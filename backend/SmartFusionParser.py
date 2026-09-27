import os
import math
import laspy
import numpy as np
import cv2  # Добавлено для морфологии маски
from PIL import Image, ImageDraw
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from .base import Tree


class SmartFusionParser:
    """
    SmartFusionParser (Ensemble V2).
    Исправленная версия с защитой от дублирования (Over-segmentation)
    и переработанной логикой слияния.
    """

    def __init__(self, input_path: str, output_img_path: str, progress_callback, error_callback):
        self.input_path = input_path
        self.output_img_path = output_img_path
        self.on_progress = progress_callback
        self.on_error = error_callback

        # --- Настройки ---
        # Очистка
        self.sor_neighbors = 8
        self.sor_std_mul = 1.5

        # Layer Stacking (Геометрия)
        self.slice_thickness = 0.5
        self.layer_eps = 0.9
        self.min_samples = 5
        self.link_dist = 2.0

        # CHM (Высота)
        self.pixel_size = 0.3
        self.smooth_sigma = 0.8
        self.min_tree_height = 2.5

        # Post-Processing
        self.merge_overlap_threshold = 0.7

        # Data
        self.clean_points = None
        self.z_ground = 0.0
        self.chm_grid = None
        self.geo_transform = {}
        self.ls_clusters = []
        self.result_trees = []

    def get_trees(self):
        return self.result_trees

    def full_parse(self):
        try:
            self._report_progress(0)
            if not self._load_data(): return

            # 1. Очистка (10-25%)
            self._denoise_data()

            # 2. CHM (25-40%)
            self._generate_chm()

            # 3. Layer Stacking (40-70%)
            self._run_layer_stacking()

            # 4. Fusion & Recovery (70-85%)
            self._fusion_and_correction()

            # 5. De-duplication (85-95%) - НОВЫЙ ЭТАП
            self._remove_duplicates()

            # 6. Визуализация (95-100%)
            self._visualize()

            self._report_progress(100)
        except Exception as e:
            if self.on_error: self.on_error(e)

    def print_report(self):
        """Печатает отчет (исправлено отсутствие метода)."""
        if not self.result_trees:
            print("Нет данных.")
            return

        counts = {}
        heights = []
        radii = []

        for t in self.result_trees:
            counts[t.shape_type] = counts.get(t.shape_type, 0) + 1
            heights.append(t.height)
            radii.append(t.radius)

        print(f"--- Отчет SmartFusion: {os.path.basename(self.input_path)} ---")
        print(f"Всего деревьев: {len(self.result_trees)}")
        for type_name, count in counts.items():
            print(f"  Тип '{type_name}': {count} шт.")
        if heights:
            print(f"  Средняя высота: {sum(heights) / len(heights):.2f} м")
            print(f"  Средний радиус: {sum(radii) / len(radii):.2f} м")

    # --- Internal ---

    def _report_progress(self, val):
        if self.on_progress: self.on_progress(float(val))

    def _load_data(self):
        if not os.path.exists(self.input_path): return False
        las = laspy.read(self.input_path)

        # Упрощенная загрузка для скорости
        points = np.vstack((las.x, las.y, las.z)).T

        # Поиск земли
        if hasattr(las, 'classification'):
            mask_g = (np.array(las.classification) == 2)
            if np.any(mask_g):
                self.z_ground = np.percentile(las.z[mask_g], 50)
            else:
                self.z_ground = np.min(las.z)
        else:
            self.z_ground = np.min(las.z)

        points[:, 2] -= self.z_ground
        self.clean_points = points[points[:, 2] >= self.min_tree_height]

        if len(self.clean_points) == 0:
            raise ValueError("No points above ground.")

        self._report_progress(10)
        return True

    def _denoise_data(self):
        # SOR Filter
        pts = self.clean_points
        if len(pts) < 100: return

        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=self.sor_neighbors, workers=-1)
        mean_d = np.mean(dists, axis=1)

        thresh = np.mean(mean_d) + self.sor_std_mul * np.std(mean_d)
        self.clean_points = pts[mean_d < thresh]
        self._report_progress(25)

    def _generate_chm(self):
        pts = self.clean_points
        x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
        y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])

        cols = int(np.ceil((x_max - x_min) / self.pixel_size)) + 1
        rows = int(np.ceil((y_max - y_min) / self.pixel_size)) + 1

        self.geo_transform = {'x_min': x_min, 'y_max': y_max, 'res': self.pixel_size, 'cols': cols, 'rows': rows}

        # Grid creation
        ix = ((pts[:, 0] - x_min) / self.pixel_size).astype(np.int32)
        iy = ((y_max - pts[:, 1]) / self.pixel_size).astype(np.int32)
        valid = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
        ix, iy, z = ix[valid], iy[valid], pts[valid, 2]

        # Fast rasterization
        idx = iy * cols + ix
        order = np.argsort(idx)
        idx_s, z_s = idx[order], z[order]

        # Max Z per pixel
        unique_idx, split_idx = np.unique(idx_s, return_index=True)
        # Using reduceat logic manually or simple loop for max
        # A simpler approach using lexsort from previous attempts was efficient enough
        # Re-implementing fast max:
        max_z_vals = np.maximum.reduceat(z_s, split_idx)

        self.chm_grid = np.zeros((rows, cols), dtype=np.float32)
        self.chm_grid.flat[unique_idx] = max_z_vals

        # Closing holes & Smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.chm_grid = cv2.morphologyEx(self.chm_grid, cv2.MORPH_CLOSE, kernel)
        self.chm_grid = gaussian_filter(self.chm_grid, sigma=self.smooth_sigma)

        self._report_progress(40)

    def _run_layer_stacking(self):
        # Стандартный LS, но на очищенных данных
        pts = self.clean_points
        z_max = np.max(pts[:, 2])
        layers = np.arange(z_max, self.min_tree_height, -self.slice_thickness)

        centroids = []

        # 1. Slicing
        total_l = len(layers)
        for i, z_top in enumerate(layers):
            z_bot = z_top - self.slice_thickness
            # Быстрый фильтр по Z
            slice_pts = pts[(pts[:, 2] <= z_top) & (pts[:, 2] > z_bot)]

            if len(slice_pts) >= self.min_samples:
                db = DBSCAN(eps=self.layer_eps, min_samples=self.min_samples).fit(slice_pts[:, :2])
                for lbl in set(db.labels_):
                    if lbl == -1: continue
                    c_pts = slice_pts[db.labels_ == lbl]
                    center = np.mean(c_pts, axis=0)
                    # XY + Z + Radius
                    rad = np.max(np.linalg.norm(c_pts[:, :2] - center[:2], axis=1))
                    centroids.append([center[0], center[1], center[2], rad])

            if i % 5 == 0: self._report_progress(40 + (i / total_l) * 20)

        if not centroids: return
        centroids = np.array(centroids)

        # 2. Linking
        # Сжимаем Z очень сильно, чтобы кластеризовать вертикально
        X_link = centroids[:, :3].copy()
        X_link[:, 2] /= 4.0

        db = DBSCAN(eps=self.link_dist, min_samples=3).fit(X_link)

        self.ls_clusters = []
        for lbl in set(db.labels_):
            if lbl == -1: continue
            self.ls_clusters.append(centroids[db.labels_ == lbl])

        self._report_progress(70)

    def _fusion_and_correction(self):
        self.result_trees = []

        # Карта занятости для Recovery. Используем uint8 для масок cv2
        occupied_mask = np.zeros(self.chm_grid.shape, dtype=np.uint8)

        # --- A. Process LS Clusters ---
        for i, nodes in enumerate(self.ls_clusters):
            # 1. Геометрия от LS
            z_vals = nodes[:, 2]
            r_vals = nodes[:, 3]

            ls_h = np.max(z_vals)
            ls_r = np.percentile(r_vals, 90)

            # Центр (верхние 20% по высоте - более надежный центр кроны)
            top_mask = z_vals > np.percentile(z_vals, 80)
            if np.sum(top_mask) == 0: top_mask = np.ones_like(z_vals, dtype=bool)
            center = np.mean(nodes[top_mask, :2], axis=0)

            # 2. Уточнение высоты по CHM
            px_c = int((center[0] - self.geo_transform['x_min']) / self.geo_transform['res'])
            px_r = int((self.geo_transform['y_max'] - center[1]) / self.geo_transform['res'])

            # Ищем пик на CHM в пределах радиуса дерева
            search_r_px = int(ls_r / self.geo_transform['res'])
            search_r_px = max(2, min(search_r_px, 30))  # Лимит окна поиска

            # Границы окна
            r1 = max(0, px_r - search_r_px)
            r2 = min(self.geo_transform['rows'], px_r + search_r_px + 1)
            c1 = max(0, px_c - search_r_px)
            c2 = min(self.geo_transform['cols'], px_c + search_r_px + 1)

            final_h = ls_h
            if (r2 > r1) and (c2 > c1):
                local_chm = self.chm_grid[r1:r2, c1:c2]
                if local_chm.size > 0:
                    chm_max = np.max(local_chm)
                    # Если CHM выше, но не абсурдно (не налет птицы), берем его
                    if chm_max > ls_h and chm_max < ls_h + 5.0:
                        final_h = float(chm_max)

            # 3. Маркировка занятости
            # Рисуем круг на маске, чтобы Recovery не нашел это дерево снова
            cv2.circle(occupied_mask, (px_c, px_r), search_r_px, 1, -1)

            # Форма
            shape = self._guess_shape(nodes, final_h)

            self.result_trees.append(Tree(i, center[0], center[1], final_h, ls_r, shape))

        # --- B. Recovery Mode (Strict) ---
        # 1. Расширяем маску занятости (Dilation), чтобы не находить ветки рядом
        kernel = np.ones((5, 5), np.uint8)  # ~2 метра при res=0.4
        dilated_mask = cv2.dilate(occupied_mask, kernel, iterations=2)

        # 2. Ищем пики только там, где маска == 0
        masked_chm = self.chm_grid.copy()
        masked_chm[dilated_mask > 0] = 0

        # Мин дистанция между пиками ~3 метра
        peaks = peak_local_max(masked_chm,
                               min_distance=int(3.0 / self.geo_transform['res']),
                               threshold_abs=self.min_tree_height)

        base_id = 100000
        for j, (r, c) in enumerate(peaks):
            h = float(masked_chm[r, c])
            x = self.geo_transform['x_min'] + c * self.geo_transform['res']
            y = self.geo_transform['y_max'] - r * self.geo_transform['res']

            # Консервативный радиус для восстановленных
            rad = max(1.5, h * 0.2)

            self.result_trees.append(Tree(base_id + j, x, y, h, rad, "spherical"))

        self._report_progress(85)

    def _remove_duplicates(self):
        """
        Жадное объединение деревьев.
        Сортируем по высоте (сверху вниз). Если дерево пересекается с более высоким
        соседом слишком сильно — удаляем его.
        """
        if not self.result_trees: return

        # Сортируем: сначала самые высокие
        sorted_trees = sorted(self.result_trees, key=lambda t: t.height, reverse=True)
        final_list = []

        # KDTree для быстрого поиска соседей
        # (Можно оптимизировать, но брутфорс по списку тоже сойдет для <1000 деревьев)
        # Для надежности используем простой проход, т.к. список меняется

        active_indices = list(range(len(sorted_trees)))
        kept_indices = []

        # Преобразуем в numpy для скорости
        positions = np.array([[t.x, t.y] for t in sorted_trees])
        radii = np.array([t.radius for t in sorted_trees])
        heights = np.array([t.height for t in sorted_trees])

        processed = np.zeros(len(sorted_trees), dtype=bool)

        for i in range(len(sorted_trees)):
            if processed[i]: continue

            # Дерево i принимается
            kept_indices.append(i)
            processed[i] = True

            # Ищем всех кандидатов на удаление (кто ниже и близко)
            # Расстояние
            dists = np.linalg.norm(positions - positions[i], axis=1)

            # Условие слияния:
            # 1. Расстояние < (R1 + R2) * overlap_factor
            # 2. При этом кандидат ниже (уже гарантировано сортировкой)

            for j in range(i + 1, len(sorted_trees)):
                if processed[j]: continue

                dist = dists[j]
                sum_r = radii[i] + radii[j]

                # Если центры ближе, чем 2 метра - почти наверняка дубль
                # ИЛИ если перекрытие крон сильное
                if dist < 2.0 or (dist < sum_r * 0.6):
                    # Поглощаем дерево j (оно меньше/ниже)
                    processed[j] = True

        # Собираем результат
        self.result_trees = [sorted_trees[i] for i in kept_indices]

    def _guess_shape(self, nodes, h_total):
        # Простая эвристика по профилю
        z = nodes[:, 2]
        r = nodes[:, 3]

        if len(z) < 5: return "spherical"

        h_rel = h_total - z
        h_rel = h_rel / (np.max(h_rel) + 0.01)
        r_rel = r / (np.max(r) + 0.01)

        # Конус: R ~ H (линейно)
        # Сфера: R ~ sqrt(H) (выпукло)

        # Сравниваем корреляцию
        corr_cone = np.corrcoef(r_rel, h_rel)[0, 1]
        corr_sphere = np.corrcoef(r_rel, np.sqrt(h_rel))[0, 1]

        return "conical" if corr_cone > corr_sphere else "spherical"

    def _visualize(self):
        res = 0.25
        pts = self.clean_points
        if pts is None or len(pts) == 0: return

        x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
        y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
        w = int((x_max - x_min) / res) + 1
        h = int((y_max - y_min) / res) + 1

        # Ограничитель
        if max(w, h) > 3000:
            res *= 2
            w = int((x_max - x_min) / res) + 1
            h = int((y_max - y_min) / res) + 1

        img = Image.new("RGB", (w, h), (20, 20, 20))
        draw = ImageDraw.Draw(img)

        # Точки
        ix = ((pts[:, 0] - x_min) / res).astype(int)
        iy = ((y_max - pts[:, 1]) / res).astype(int)
        valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)

        pixels = img.load()
        # Медленно, но просто. Для продакшена нужен numpy буфер, но PIL надежнее по зависимостям
        # Для скорости рисуем только деревья

        for t in self.result_trees:
            cx = (t.x - x_min) / res
            cy = (y_max - t.y) / res
            cr = t.radius / res

            col = (255, 50, 50) if t.shape_type == 'conical' else (255, 200, 0)
            if t.id >= 100000: col = (0, 200, 255)  # Cyan

            draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], outline=col, width=2)
            draw.text((cx, cy), f"{t.height:.1f}", fill="white")

        img.save(self.output_img_path)