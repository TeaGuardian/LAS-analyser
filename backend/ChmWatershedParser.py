import os
import cv2
import laspy
import numpy as np
from .base import Tree
from PIL import Image, ImageDraw
from skimage.measure import regionprops
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


class ChmWatershedParser:
    """
    Парсер, реализующий алгоритм CHM-Watershed (Computer Vision).
    Этапы: Растеризация (CHM) -> Сглаживание -> Поиск пиков -> Водораздел -> Экстракция.
    """

    def __init__(self, input_path: str, output_img_path: str, progress_callback, error_callback):
        """
        :param input_path: Полный путь к .las/.laz файлу.
        :param output_img_path: Полный путь для сохранения PNG.
        :param progress_callback: Функция func(float), принимающая прогресс 0.0-100.0.
        :param error_callback: Функция func(Exception), принимающая ошибку.
        """
        self.input_path = input_path
        self.output_img_path = output_img_path
        self.on_progress = progress_callback
        self.on_error = error_callback

        # Параметры алгоритма
        self.pixel_size = 0.5  # Размер пикселя (GSD) в метрах
        self.smooth_sigma = 1.0  # Сила сглаживания (Gaussian sigma)
        self.min_tree_height = 2.0  # Мин. высота дерева

        # Внутренние данные
        self.points = None
        self.z_ground = 0.0
        self.chm_grid = None  # Растр высот
        self.labels_grid = None  # Растр сегментации
        self.geo_transform = {}  # Данные для перевода пикселей в метры
        self.result_trees = []

    def get_trees(self):
        """Возвращает список найденных деревьев (экземпляры класса Tree)."""
        return self.result_trees

    def full_parse(self):
        """Запускает полный цикл анализа."""
        try:
            self._update_progress(0)

            # 1. Загрузка и фильтрация (0-15%)
            if not self._load_and_filter():
                return

            # 2. Генерация CHM (Canopy Height Model) (15-40%)
            self._generate_chm()

            # 3. Сегментация Watershed (40-70%)
            self._run_watershed()

            # 4. Экстракция параметров деревьев (70-90%)
            self._extract_trees_from_segments()

            # 5. Визуализация (90-100%)
            self._visualize()

            self._update_progress(100.0)

        except Exception as e:
            self.on_error(e)

    def print_report(self):
        """Печатает краткую статистику."""
        if not self.result_trees:
            print("Нет данных для отчета.")
            return

        counts = {}
        heights = []
        radii = []

        for t in self.result_trees:
            counts[t.shape_type] = counts.get(t.shape_type, 0) + 1
            heights.append(t.height)
            radii.append(t.radius)

        print(f"--- Отчет CHM-Watershed: {os.path.basename(self.input_path)} ---")
        print(f"Всего деревьев: {len(self.result_trees)}")
        for type_name, count in counts.items():
            print(f"  Тип '{type_name}': {count} шт.")
        if heights:
            print(f"  Средняя высота: {sum(heights) / len(heights):.2f} м")
            print(f"  Средний радиус: {sum(radii) / len(radii):.2f} м")

    # --- Внутренние методы ---

    def _update_progress(self, val):
        if self.on_progress:
            self.on_progress(float(val))

    def _load_and_filter(self):
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Файл не найден: {self.input_path}")

        try:
            las = laspy.read(self.input_path)

            # Стандартная логика фильтрации
            if hasattr(las, 'classification'):
                raw_cls = np.array(las.classification)
                ground_mask = (raw_cls == 2)

                if np.any(ground_mask):
                    self.z_ground = np.percentile(np.array(las.z)[ground_mask], 50)
                else:
                    self.z_ground = np.min(las.z)

                veg_classes = (3, 4, 5)
                mask_veg = np.isin(raw_cls, veg_classes)

                if np.sum(mask_veg) == 0:
                    self.points = np.vstack((las.x, las.y, las.z)).T
                    self.points[:, 2] -= self.z_ground
                    self.points = self.points[self.points[:, 2] >= self.min_tree_height]
                else:
                    self.points = np.vstack((
                        las.x[mask_veg], las.y[mask_veg], las.z[mask_veg]
                    )).T
                    self.points[:, 2] -= self.z_ground
            else:
                self.points = np.vstack((las.x, las.y, las.z)).T
                self.z_ground = np.min(self.points[:, 2])
                self.points[:, 2] -= self.z_ground
                self.points = self.points[self.points[:, 2] >= self.min_tree_height]

            if len(self.points) == 0:
                raise ValueError("Точек для анализа не найдено.")

            self._update_progress(15.0)
            return True
        except Exception as e:
            self.on_error(e)
            return False

    def _generate_chm(self):
        """Создает растр высот."""
        x_min, x_max = np.min(self.points[:, 0]), np.max(self.points[:, 0])
        y_min, y_max = np.min(self.points[:, 1]), np.max(self.points[:, 1])

        cols = int(np.ceil((x_max - x_min) / self.pixel_size)) + 1
        rows = int(np.ceil((y_max - y_min) / self.pixel_size)) + 1

        # Сохраняем гео-трансформ для обратного перевода
        self.geo_transform = {
            'x_min': x_min, 'y_max': y_max,
            'cols': cols, 'rows': rows,
            'res': self.pixel_size
        }

        # Векторизованная растеризация (быстрый поиск макс Z в ячейке)
        # Y инвертируем (изображение: 0 сверху)
        ix = ((self.points[:, 0] - x_min) / self.pixel_size).astype(np.int32)
        iy = ((y_max - self.points[:, 1]) / self.pixel_size).astype(np.int32)

        valid = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
        ix, iy, z_vals = ix[valid], iy[valid], self.points[valid, 2]

        # Lexsort: сортируем по индексу пикселя, затем по высоте
        pixel_idx = iy * cols + ix
        sort_idx = np.lexsort((z_vals, pixel_idx))

        sorted_pix = pixel_idx[sort_idx]
        sorted_z = z_vals[sort_idx]
        unique_pix = np.unique(sorted_pix)

        # Берем последнее вхождение (максимальное Z) для каждого пикселя
        max_idx = np.searchsorted(sorted_pix, unique_pix, side='right') - 1

        self.chm_grid = np.zeros((rows, cols), dtype=np.float32)
        self.chm_grid[unique_pix // cols, unique_pix % cols] = sorted_z[max_idx]

        # Закрытие дыр (morphology close) и сглаживание
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.chm_grid = cv2.morphologyEx(self.chm_grid, cv2.MORPH_CLOSE, kernel)

        # Сглаживание критично для watershed, чтобы не было over-segmentation
        self.chm_grid = gaussian_filter(self.chm_grid, sigma=self.smooth_sigma)

        self._update_progress(40.0)

    def _run_watershed(self):
        """Запускает сегментацию."""
        # 1. Поиск локальных максимумов (вершины)
        local_maxi = peak_local_max(
            self.chm_grid,
            min_distance=int(2.0 / self.pixel_size),  # Мин расстояние между вершинами ~2м
            threshold_abs=self.min_tree_height,
            labels=self.chm_grid > 0  # Маска: где есть данные
        )

        # 2. Создание маркеров
        markers = np.zeros_like(self.chm_grid, dtype=int)
        for i, (r, c) in enumerate(local_maxi):
            markers[r, c] = i + 1

        # 3. Водораздел
        # Watershed работает на "заполнение бассейнов". Инвертируем высоту.
        mask = self.chm_grid > self.min_tree_height
        self.labels_grid = watershed(-self.chm_grid, markers, mask=mask)

        self._update_progress(70.0)

    def _extract_trees_from_segments(self):
        """Анализ свойств сегментов (regionprops) и создание Tree."""
        self.result_trees = []

        # Используем skimage regionprops для быстрого анализа свойств регионов
        props = regionprops(self.labels_grid, intensity_image=self.chm_grid)
        total_regions = len(props)

        for i, region in enumerate(props):
            # region.label - ID
            # region.intensity_max - Максимальная высота
            # region.centroid - Центр масс (Y, X)
            # region.equivalent_diameter - Диаметр круга той же площади

            h = region.intensity_max
            if h < self.min_tree_height: continue

            # Координаты
            r_center, c_center = region.centroid
            # Конвертация в метры
            # X = x_min + col * res
            # Y = y_max - row * res
            x_real = self.geo_transform['x_min'] + c_center * self.geo_transform['res']
            y_real = self.geo_transform['y_max'] - r_center * self.geo_transform['res']

            radius = (region.equivalent_diameter_area / 2.0) * self.geo_transform['res']

            # Определение формы (Эвристика для 2.5D)
            # Сравниваем макс высоту и среднюю высоту региона.
            # Конус: Mean ~ 1/3 Max (объем конуса).
            # Сфера/Цилиндр: Mean ближе к Max.
            shape = "spherical"
            if region.intensity_mean < (h * 0.6):
                shape = "conical"

            tree = Tree(
                tree_id=region.label,
                x=x_real,
                y=y_real,
                height=float(h),
                radius=float(radius),
                shape_type=shape
            )
            self.result_trees.append(tree)

            if i % 100 == 0:
                self._update_progress(70 + (i / total_regions) * 20)

    def _visualize(self):
        """Генерация PNG карты на основе CHM."""
        resolution = 0.2  # Выходное разрешение картинки

        x_min = self.geo_transform['x_min']
        y_max = self.geo_transform['y_max']

        # Используем размеры исходного CHM для простоты, масштабируем при сохранении
        # Но лучше пересоздать канвас, чтобы унифицировать с другими парсерами
        width_m = self.geo_transform['cols'] * self.geo_transform['res']
        height_m = self.geo_transform['rows'] * self.geo_transform['res']

        # Ограничение размера 2048
        MAX_DIM = 2048
        w_px = width_m / resolution
        h_px = height_m / resolution

        if w_px > MAX_DIM or h_px > MAX_DIM:
            res_w = width_m / MAX_DIM
            res_h = height_m / MAX_DIM
            resolution = max(res_w, res_h) * 1.001

        img_w = int(np.ceil(width_m / resolution))
        img_h = int(np.ceil(height_m / resolution))

        # Создаем фон из CHM (интерполяция)
        # Нормализуем CHM для цвета
        chm_norm = (self.chm_grid - np.min(self.chm_grid)) / (np.max(self.chm_grid) - np.min(self.chm_grid) + 1e-6)

        # Ресайз CHM до размера output картинки
        # chm_grid (float 0..1) -> uint8 image -> resize -> numpy
        chm_uint8 = (chm_norm * 255).astype(np.uint8)
        pil_chm = Image.fromarray(chm_uint8, mode='L')
        pil_chm = pil_chm.resize((img_w, img_h), resample=Image.BILINEAR)

        # Раскраска (Terrain)
        chm_resized = np.array(pil_chm) / 255.0
        try:
            import matplotlib.cm as cm
            cmap = cm.get_cmap('terrain')
            # Убираем альфа-канал
            canvas = (cmap(chm_resized)[:, :, :3] * 255).astype(np.uint8)
        except:
            canvas = np.stack((chm_uint8, chm_uint8, chm_uint8), axis=2)

        img = Image.fromarray(canvas, 'RGB')
        draw = ImageDraw.Draw(img)

        # Отрисовка деревьев
        for t in self.result_trees:
            px = (t.x - x_min) / resolution
            py = (y_max - t.y) / resolution
            r_px = max(2.0, t.radius / resolution)

            if t.shape_type == 'conical':
                outline_col = (255, 30, 30)
            elif t.shape_type == 'spherical':
                outline_col = (255, 215, 0)
            else:
                outline_col = (100, 200, 255)

            lw = max(1, int(2.0 / resolution * 0.15))
            draw.ellipse([px - r_px, py - r_px, px + r_px, py + r_px], outline=outline_col, width=lw)

            dot_sz = max(1, 0.3 / resolution)
            draw.ellipse([px - dot_sz, py - dot_sz, px + dot_sz, py + dot_sz], fill=(255, 255, 255))

        img.save(self.output_img_path)