import os
import laspy
import numpy as np
from .base import Tree
from PIL import Image, ImageDraw
from sklearn.cluster import DBSCAN


class LayerStackParser:
    """
    Парсер, реализующий алгоритм Layer-Stacking (Slice-based DBSCAN).
    Этапы: Slicing -> 2D DBSCAN per slice -> Vertical Linking -> Tree Construction.
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
        self.slice_thickness = 0.5  # Толщина слоя (м)
        self.layer_eps = 0.8  # Радиус поиска DBSCAN внутри слоя (м)
        self.min_samples = 4  # Мин. точек для кластера в слое
        self.link_dist = 1.5  # Макс. вертикальное расстояние для связывания слоев
        self.min_tree_height = 2.0  # Мин. высота дерева

        # Внутренние данные
        self.points = None  # Numpy array [x, y, z]
        self.z_ground = 0.0
        self.slice_centroids = []  # Центры кластеров в слоях [x, y, z, r]
        self.result_trees = []  # Список экземпляров класса Tree

    def get_trees(self):
        """Возвращает список найденных деревьев (экземпляры класса Tree)."""
        return self.result_trees

    def full_parse(self):
        """Запускает полный цикл анализа."""
        try:
            self._update_progress(0)

            # 1. Загрузка и фильтрация (0-20%)
            if not self._load_and_filter():
                return

            # 2. Нарезка на слои и 2D кластеризация (20-60%)
            self._slice_and_cluster()

            # 3. Вертикальное связывание (Linking) (60-80%)
            tree_candidates = self._link_vertical()

            # 4. Анализ формы и создание объектов Tree (80-90%)
            self._analyze_candidates(tree_candidates)

            # 5. Визуализация (90-100%)
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

        print(f"--- Отчет Layer-Stacking: {os.path.basename(self.input_path)} ---")
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

            # Извлекаем классификацию и высоту
            if hasattr(las, 'classification'):
                raw_cls = np.array(las.classification)
                ground_mask = (raw_cls == 2)

                if np.any(ground_mask):
                    self.z_ground = np.percentile(np.array(las.z)[ground_mask], 50)
                else:
                    self.z_ground = np.min(las.z)

                # Берем растительность (3, 4, 5)
                veg_classes = (3, 4, 5)
                mask_veg = np.isin(raw_cls, veg_classes)

                if np.sum(mask_veg) == 0:
                    # Fallback: берем все точки выше земли + min_h
                    self.points = np.vstack((las.x, las.y, las.z)).T
                    self.points[:, 2] -= self.z_ground
                    self.points = self.points[self.points[:, 2] >= self.min_tree_height]
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
                self.points = self.points[self.points[:, 2] >= self.min_tree_height]

            if len(self.points) == 0:
                raise ValueError("После фильтрации нет точек.")

            self._update_progress(20.0)
            return True

        except Exception as e:
            self.on_error(e)
            return False

    def _slice_and_cluster(self):
        """Проходит по слоям и выполняет DBSCAN для каждого."""
        if len(self.points) == 0: return

        z_max = np.max(self.points[:, 2])
        z_min = np.min(self.points[:, 2])

        layers = np.arange(z_max, z_min, -self.slice_thickness)
        total_layers = len(layers)

        self.slice_centroids = []

        for i, z_top in enumerate(layers):
            z_bot = z_top - self.slice_thickness

            # Выбираем точки слоя
            mask = (self.points[:, 2] <= z_top) & (self.points[:, 2] > z_bot)
            pts_layer = self.points[mask]

            if len(pts_layer) > self.min_samples:
                # DBSCAN только по XY
                xy = pts_layer[:, :2]

                # eps - радиус поиска соседей
                db = DBSCAN(eps=self.layer_eps, min_samples=self.min_samples).fit(xy)
                labels = db.labels_

                unique_labels = set(labels)
                if -1 in unique_labels: unique_labels.remove(-1)

                for lbl in unique_labels:
                    cluster_mask = (labels == lbl)
                    cluster_pts = pts_layer[cluster_mask]

                    # Центр масс
                    center = np.mean(cluster_pts, axis=0)  # [x, y, z]

                    # Радиус (макс удаление от центра в плоскости XY)
                    dists = np.linalg.norm(cluster_pts[:, :2] - center[:2], axis=1)
                    radius = np.max(dists)

                    # Сохраняем узел: x, y, z, r
                    self.slice_centroids.append([center[0], center[1], center[2], radius])

            # Прогресс 20-60%
            if i % 10 == 0:
                self._update_progress(20 + (i / total_layers) * 40)

        self.slice_centroids = np.array(self.slice_centroids)

    def _link_vertical(self):
        """Связывает центроиды слоев в деревья."""
        if len(self.slice_centroids) == 0: return []

        # Используем DBSCAN на центроидах слоев
        # Сжимаем Z, чтобы объединять преимущественно вертикально
        X = self.slice_centroids[:, :3].copy()

        # Масштабный коэффициент для Z.
        # Если Z не сжать, DBSCAN объединит только очень близкие слои.
        # Нам нужно, чтобы он "прыгал" на link_dist вверх.
        # При eps=link_dist, расстояние по Z должно быть значимым.
        # Просто используем метрику где Z весит меньше (или используем link_dist как eps)

        # Простой подход: DBSCAN в 3D с eps = link_dist
        # Но Z обычно больше шага по XY.
        # Используем веса: X, Y важнее.
        X[:, 2] /= 2.0  # Сжимаем высоту

        db = DBSCAN(eps=self.link_dist, min_samples=3).fit(X)
        labels = db.labels_

        unique_trees = set(labels)
        if -1 in unique_trees: unique_trees.remove(-1)

        candidates = []
        for tid in unique_trees:
            mask = (labels == tid)
            nodes = self.slice_centroids[mask]
            candidates.append(nodes)

        self._update_progress(80.0)
        return candidates

    def _analyze_candidates(self, candidates):
        """Превращает списки узлов в объекты Tree."""
        self.result_trees = []
        total = len(candidates)

        for i, nodes in enumerate(candidates):
            # nodes: [[x, y, z, r], ...]

            # Высота
            z_vals = nodes[:, 2]
            z_max = np.max(z_vals)
            z_min = np.min(z_vals)
            height = z_max - z_min  # Т.к. Z уже нормализован относительно земли

            if height < self.min_tree_height:
                continue

            # Радиус (берем 90-й перцентиль радиусов слоев или макс радиус самого широкого слоя)
            # Лучше взять средний макс радиус
            radii = nodes[:, 3]
            max_r = np.percentile(radii, 90)

            # Вершина (центр самого верхнего слоя)
            top_idx = np.argmax(z_vals)
            top_node = nodes[top_idx]
            top_x, top_y = top_node[0], top_node[1]

            # Определение формы
            shape = self._fit_shape(nodes, z_max)

            # Создаем объект
            tree = Tree(
                tree_id=i,  # Временный ID
                x=top_x,
                y=top_y,
                height=float(z_max),  # Высота от земли
                radius=float(max_r),
                shape_type=shape
            )
            self.result_trees.append(tree)

            if i % 20 == 0:
                self._update_progress(80 + (i / total) * 10)

    def _fit_shape(self, nodes, h_total):
        """Апроксимация формы по узлам слоев."""
        # Данные: h (расстояние от вершины), r (радиус слоя)
        z_vals = nodes[:, 2]
        r_vals = nodes[:, 3]

        h_rel = h_total - z_vals  # 0 в вершине

        # Фильтруем (иногда радиусы шумят)
        if len(h_rel) < 3: return "undefined"

        # Модели
        def cone_err(k): return np.sum((r_vals - k * h_rel) ** 2)

        def ball_err(k): return np.sum((r_vals - k * np.sqrt(np.abs(h_rel))) ** 2)

        # Оценка k
        k_cone = np.sum(r_vals * h_rel) / (np.sum(h_rel ** 2) + 1e-6)
        k_ball = np.sum(r_vals * np.sqrt(np.abs(h_rel))) / (np.sum(h_rel) + 1e-6)

        err_c = cone_err(k_cone)
        err_b = ball_err(k_ball)

        return "conical" if err_c < err_b else "spherical"

    def _visualize(self):
        """Генерация PNG карты."""
        if len(self.points) == 0: return

        resolution = 0.2

        x_min, x_max = np.min(self.points[:, 0]), np.max(self.points[:, 0])
        y_min, y_max = np.min(self.points[:, 1]), np.max(self.points[:, 1])

        width_m = x_max - x_min
        height_m = y_max - y_min

        # Ограничение размера
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

        # Точки
        ix = ((self.points[:, 0] - x_min) / resolution).astype(int)
        iy = ((y_max - self.points[:, 1]) / resolution).astype(int)
        ix = np.clip(ix, 0, img_w - 1)
        iy = np.clip(iy, 0, img_h - 1)

        z_vals = self.points[:, 2]
        z_norm = (z_vals - np.min(z_vals)) / (np.max(z_vals) - np.min(z_vals) + 1e-6)

        # Цвета
        try:
            import matplotlib.cm as cm
            cmap = cm.get_cmap('terrain')
            colors = (cmap(z_norm)[:, :3] * 255).astype(np.uint8)
        except:
            colors = np.zeros((len(z_norm), 3), dtype=np.uint8)
            colors[:, 1] = (z_norm * 255).astype(np.uint8)

        sort_order = np.argsort(z_vals)
        canvas[iy[sort_order], ix[sort_order]] = colors[sort_order]

        img = Image.fromarray(canvas, 'RGB')
        draw = ImageDraw.Draw(img)

        # Деревья
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