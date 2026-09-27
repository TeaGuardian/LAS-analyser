import os
import laspy
import numpy as np
from .base import Tree
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max


class CrownAxisParser:
    """
    Парсер, реализующий алгоритм CROWN-AXIS (Hybrid 3D Seeding).
    Этапы: CHM Seeds -> Axis Descent -> Anisotropic Competition -> Shape Analysis.
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
        self.veg_classes = (3, 4, 5)  # Low, Medium, High vegetation
        self.min_z = 2.0  # Мин. высота дерева
        self.axis_step = 1.0  # Шаг спуска оси
        self.search_radius = 2.5  # Радиус поиска оси

        # Внутренние данные
        self.points = None  # Numpy array [x, y, z]
        self.z_ground = 0.0
        self.peak_points = []  # Вершины
        self.axes = []  # Оси
        self.point_labels = None  # ID дерева для каждой точки
        self.result_trees = []  # Список экземпляров класса Tree

    def get_trees(self):
        """Возвращает список найденных деревьев (экземпляры класса Tree)."""
        return self.result_trees

    def full_parse(self):
        """Запускает полный цикл анализа."""
        try:
            self._update_progress(0)

            # 1. Загрузка и фильтрация (0-15%)
            if not self._load_and_filter():
                return  # Ошибка уже отправлена внутри

            # 2. Поиск вершин (15-30%)
            self._find_peaks()

            # 3. Построение осей (30-50%)
            self._build_axes()

            # 4. Конкуренция осей (50-75%)
            self._assign_points()

            # 5. Анализ форм и создание объектов Tree (75-90%)
            self._analyze_clusters()

            # 6. Визуализация (90-100%)
            self._visualize()

            self._update_progress(100.0)

        except Exception as e:
            self.on_error(e)

    def print_report(self):
        """Печатает краткую статистику в консоль."""
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

        print(f"--- Отчет CROWN-AXIS: {os.path.basename(self.input_path)} ---")
        print(f"Всего деревьев: {len(self.result_trees)}")
        for type_name, count in counts.items():
            print(f"  Тип '{type_name}': {count} шт.")
        if heights:
            print(f"  Средняя высота: {sum(heights) / len(heights):.2f} м")
            print(f"  Средний радиус: {sum(radii) / len(radii):.2f} м")

    # --- Внутренние методы ---

    def _update_progress(self, val):
        """Безопасный вызов коллбека прогресса."""
        if self.on_progress:
            self.on_progress(float(val))

    def _load_and_filter(self):
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Файл не найден: {self.input_path}")

        try:
            las = laspy.read(self.input_path)

            # Извлекаем классификацию
            if hasattr(las, 'classification'):
                raw_cls = np.array(las.classification)
                ground_mask = (raw_cls == 2)

                # Уровень земли
                if np.any(ground_mask):
                    self.z_ground = np.percentile(np.array(las.z)[ground_mask], 50)
                else:
                    self.z_ground = np.min(las.z)

                # Фильтр растительности
                mask_veg = np.isin(raw_cls, self.veg_classes)
                if np.sum(mask_veg) == 0:
                    # Если нет классов 3-5, пробуем взять все, что выше земли + min_z
                    # Это fallback для "плохих" файлов
                    self.points = np.vstack((las.x, las.y, las.z)).T
                    self.points[:, 2] -= self.z_ground
                    self.points = self.points[self.points[:, 2] >= self.min_z]
                else:
                    self.points = np.vstack((
                        las.x[mask_veg], las.y[mask_veg], las.z[mask_veg]
                    )).T
                    self.points[:, 2] -= self.z_ground
            else:
                # Нет классификации
                self.points = np.vstack((las.x, las.y, las.z)).T
                self.z_ground = np.min(self.points[:, 2])
                self.points[:, 2] -= self.z_ground
                self.points = self.points[self.points[:, 2] >= self.min_z]

            if len(self.points) == 0:
                raise ValueError("После фильтрации не осталось точек для анализа.")

            self._update_progress(15.0)
            return True

        except Exception as e:
            self.on_error(e)
            return False

    def _find_peaks(self):
        if len(self.points) == 0: return

        # Параметры для CHM
        res = 0.5
        x_min, y_min = np.min(self.points[:, 0]), np.min(self.points[:, 1])
        x_max, y_max = np.max(self.points[:, 0]), np.max(self.points[:, 1])

        cols = int((x_max - x_min) / res) + 1
        rows = int((y_max - y_min) / res) + 1

        # Быстрое построение CHM
        ix = ((self.points[:, 0] - x_min) / res).astype(np.int32)
        iy = ((y_max - self.points[:, 1]) / res).astype(np.int32)

        valid = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
        ix, iy, z_vals = ix[valid], iy[valid], self.points[valid, 2]

        pixel_idx = iy * cols + ix
        sort_idx = np.lexsort((z_vals, pixel_idx))

        sorted_pix = pixel_idx[sort_idx]
        sorted_z = z_vals[sort_idx]
        unique_pix = np.unique(sorted_pix)
        max_idx = np.searchsorted(sorted_pix, unique_pix, side='right') - 1

        chm = np.zeros((rows, cols), dtype=np.float32)
        chm[unique_pix // cols, unique_pix % cols] = sorted_z[max_idx]

        # Сглаживание и поиск пиков
        chm = gaussian_filter(chm, sigma=1.5)
        local_maxi = peak_local_max(chm, min_distance=int(2.5 / res), threshold_abs=self.min_z)

        # Уточнение в 3D
        self.peak_points = []
        self.kd_tree = cKDTree(self.points)

        total_peaks = len(local_maxi)
        for i, (r, c) in enumerate(local_maxi):
            wx = x_min + c * res
            wy = y_max - r * res
            wz = chm[r, c]
            d, idx = self.kd_tree.query([wx, wy, wz], k=15)

            neighbors = self.points[idx] if isinstance(idx, np.ndarray) else self.points[[idx]]
            if len(neighbors) > 0:
                best_idx = np.argmax(neighbors[:, 2])
                self.peak_points.append(neighbors[best_idx])

            # Прогресс внутри этапа 15-30%
            if i % 50 == 0:
                self._update_progress(15 + (i / total_peaks) * 15)

    def _build_axes(self):
        self.axes = []
        total = len(self.peak_points)

        for i, start_node in enumerate(self.peak_points):
            current_node = start_node.copy()
            axis_nodes = [current_node]

            while current_node[2] > (self.min_z / 2):
                z_target = current_node[2] - self.axis_step
                idxs = self.kd_tree.query_ball_point(current_node, r=self.search_radius)

                if not idxs: break

                neighbors = self.points[idxs]
                mask_down = (neighbors[:, 2] < current_node[2]) & (neighbors[:, 2] > z_target - 1.0)
                candidates = neighbors[mask_down]

                if len(candidates) == 0:
                    next_node = np.array([current_node[0], current_node[1], z_target])
                else:
                    next_node = np.mean(candidates, axis=0)

                axis_nodes.append(next_node)
                current_node = next_node
                if current_node[2] <= 0.5: break

            self.axes.append(np.array(axis_nodes))

            # Прогресс 30-50%
            if i % 50 == 0:
                self._update_progress(30 + (i / total) * 20)

    def _assign_points(self):
        self.point_labels = np.full(len(self.points), -1, dtype=np.int32)

        axis_pts = []
        axis_ids = []

        for tid, nodes in enumerate(self.axes):
            if len(nodes) < 2: continue
            # Интерполяция осей
            for k in range(len(nodes) - 1):
                p1, p2 = nodes[k], nodes[k + 1]
                dist = np.linalg.norm(p1 - p2)
                steps = max(1, int(dist / 0.5))
                for t in np.linspace(0, 1, steps + 1):
                    axis_pts.append(p1 * (1 - t) + p2 * t)
                    axis_ids.append(tid)

        if not axis_pts: return

        axis_pts = np.array(axis_pts)
        axis_ids = np.array(axis_ids)

        # Сжатие Z (Анизотропия)
        beta = 0.3
        axis_pts[:, 2] *= beta
        axis_tree = cKDTree(axis_pts)

        pts_scaled = self.points.copy()
        pts_scaled[:, 2] *= beta

        chunk_sz = 50000
        total_pts = len(pts_scaled)

        for i in range(0, total_pts, chunk_sz):
            _, idxs = axis_tree.query(pts_scaled[i:i + chunk_sz], k=1, workers=-1)
            self.point_labels[i:i + chunk_sz] = axis_ids[idxs]

            # Прогресс 50-75%
            self._update_progress(50 + (i / total_pts) * 25)

    def _analyze_clusters(self):
        unique_ids = np.unique(self.point_labels)
        unique_ids = unique_ids[unique_ids != -1]
        self.result_trees = []

        total = len(unique_ids)

        for i, uid in enumerate(unique_ids):
            mask = (self.point_labels == uid)
            if np.sum(mask) < 30: continue

            pts = self.points[mask]
            z_max = np.max(pts[:, 2])

            if z_max < self.min_z: continue

            # Определяем параметры
            top_idx = np.argmax(pts[:, 2])
            top = pts[top_idx]  # Относительные координаты

            center_xy = np.mean(pts[:, :2], axis=0)
            radius = np.percentile(np.linalg.norm(pts[:, :2] - center_xy, axis=1), 90)
            shape = self._fit_shape(pts, top, z_max)

            # Абсолютные координаты (x, y)
            # top[0], top[1] - это X и Y

            # Создаем экземпляр класса Tree
            tree_obj = Tree(
                tree_id=int(uid),
                x=top[0],
                y=top[1],
                height=float(z_max),
                radius=float(radius),
                shape_type=shape
            )
            self.result_trees.append(tree_obj)

            # Прогресс 75-90%
            if i % 50 == 0:
                self._update_progress(75 + (i / total) * 15)

    def _fit_shape(self, points, top, h):
        dz = top[2] - points[:, 2]
        dr = np.linalg.norm(points[:, :2] - top[:2], axis=1)
        bins = np.linspace(0, h, 8)
        r_prof, h_prof = [], []

        for k in range(len(bins) - 1):
            m = (dz >= bins[k]) & (dz < bins[k + 1])
            if np.sum(m) > 3:
                r_prof.append(np.percentile(dr[m], 85))
                h_prof.append(np.mean([bins[k], bins[k + 1]]))

        if len(h_prof) < 3: return "undefined"
        ha, ra = np.array(h_prof), np.array(r_prof)

        k_cone = np.sum(ra * ha) / np.sum(ha ** 2)
        k_ball = np.sum(ra * np.sqrt(ha)) / np.sum(ha)
        err_c = np.sum((ra - k_cone * ha) ** 2)
        err_b = np.sum((ra - k_ball * np.sqrt(ha)) ** 2)

        return "conical" if err_c < err_b else "spherical"

    def _visualize(self):
        if len(self.points) == 0: return

        resolution = 0.2

        # Границы
        x_min, x_max = np.min(self.points[:, 0]), np.max(self.points[:, 0])
        y_min, y_max = np.min(self.points[:, 1]), np.max(self.points[:, 1])

        width_m = x_max - x_min
        height_m = y_max - y_min

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

        # Фон
        canvas = np.full((img_h, img_w, 3), 20, dtype=np.uint8)

        ix = ((self.points[:, 0] - x_min) / resolution).astype(int)
        iy = ((y_max - self.points[:, 1]) / resolution).astype(int)
        ix = np.clip(ix, 0, img_w - 1)
        iy = np.clip(iy, 0, img_h - 1)

        # Цвета (Terrain)
        z_vals = self.points[:, 2]
        z_min_val, z_max_val = np.min(z_vals), np.max(z_vals)
        if z_max_val > z_min_val:
            z_norm = (z_vals - z_min_val) / (z_max_val - z_min_val)
        else:
            z_norm = np.zeros_like(z_vals)

        try:
            import matplotlib.cm as cm
            cmap = cm.get_cmap('terrain')
            colors = (cmap(z_norm)[:, :3] * 255).astype(np.uint8)
        except:
            colors = np.zeros((len(z_norm), 3), dtype=np.uint8)
            colors[:, 1] = (z_norm * 255).astype(np.uint8)

        # Отрисовка точек
        sort_order = np.argsort(z_vals)
        canvas[iy[sort_order], ix[sort_order]] = colors[sort_order]

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

            line_width = max(1, int(2.0 / resolution * 0.15))
            draw.ellipse([px - r_px, py - r_px, px + r_px, py + r_px], outline=outline_col, width=line_width)

            dot_sz = max(1, 0.3 / resolution)
            draw.ellipse([px - dot_sz, py - dot_sz, px + dot_sz, py + dot_sz], fill=(255, 255, 255))

        img.save(self.output_img_path)
        self._update_progress(100.0)
