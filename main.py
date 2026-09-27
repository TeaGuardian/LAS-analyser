import os
import laspy
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops


class Tree:
    def __init__(self, tree_id: int, x: float, y: float, height: float, radius: float, shape_type: str):
        """
        Инициализация дерева.
        :param tree_id: Уникальный ID дерева
        :param x: Координата X
        :param y: Координата Y
        :param height: Высота дерева
        :param radius: Радиус/размер кроны
        :param shape_type: Тип кроны (например, 'conical', 'spherical')
        """
        self.id = tree_id
        self.x = float(x)
        self.y = float(y)
        self.height = float(height)
        self.radius = float(radius)
        self.shape_type = str(shape_type)

    def __eq__(self, other):
        """
        Сравнение двух деревьев.
        Возвращает коэффициент схожести от 0.0 до 1.0.
        Веса:
        - Тип: 0.25
        - Высота: 0.25
        - Радиус: 0.25
        - Координаты (XY): 0.25
        """
        if not isinstance(other, Tree):
            return 0.0

        # 1. Схожесть по Типу (Type) - 25%
        # Строгое сравнение: если типы равны = 1, иначе 0
        sim_type = 1.0 if self.shape_type == other.shape_type else 0.0

        # 2. Схожесть по Высоте (Height) - 25%
        # Относительное сходство: 1 - (разница / макс_высоту)
        max_h = max(self.height, other.height)
        if max_h == 0:
            sim_height = 1.0  # Оба 0
        else:
            sim_height = max(0.0, 1.0 - abs(self.height - other.height) / max_h)

        # 3. Схожесть по Радиусу (Radius) - 25%
        max_r = max(self.radius, other.radius)
        if max_r == 0:
            sim_radius = 1.0
        else:
            sim_radius = max(0.0, 1.0 - abs(self.radius - other.radius) / max_r)

        # 4. Схожесть по Координатам (XY) - 25%
        # Евклидово расстояние
        dist = math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
        # Функция схожести: 1 / (1 + расстояние).
        # Если расстояние 0 -> 1.0. Если расстояние 1м -> 0.5. Если далеко -> стремится к 0.
        sim_pos = 1.0 / (1.0 + dist)

        # Итоговый подсчет с весами 0.25
        total_score = (sim_type * 0.25) + \
                      (sim_height * 0.25) + \
                      (sim_radius * 0.25) + \
                      (sim_pos * 0.25)

        return total_score

    def compare_with(self, other):
        if not isinstance(other, Tree):
            return 0, 0, 0, 0
        sim_type = 1.0 if self.shape_type == other.shape_type else 0.0
        max_h = max(self.height, other.height)
        if max_h == 0:
            sim_height = 1.0  # Оба 0
        else:
            sim_height = max(0.0, 1.0 - abs(self.height - other.height) / max_h)
        max_r = max(self.radius, other.radius)
        if max_r == 0:
            sim_radius = 1.0
        else:
            sim_radius = max(0.0, 1.0 - abs(self.radius - other.radius) / max_r)
        dist = math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
        sim_pos = 1.0 / (1.0 + dist)

        return sim_type, sim_height, sim_radius, sim_pos

    def __repr__(self):
        return (f"Tree(id={self.id}, type='{self.shape_type}', "
                f"h={self.height:.1f}, r={self.radius:.1f}, "
                f"pos=({self.x:.1f}, {self.y:.1f}))")


class AdaptiveFusionParser:
    """
    AdaptiveFusionParser V17 (Spectral Hunter).

    Стратегия для поиска "скрытых" и "размытых" деревьев:
    1. Dual-CHM: Пики ищутся на "резкой" карте (sigma=0.3), а границы строятся по "гладкой".
       Это позволяет увидеть микро-перепады высот.
    2. Verticality Filter: Чтобы отличить маленькое дерево от ветки большого,
       мы проверяем стандартное отклонение Z (std_z). Ветки плоские, деревья объемные.
    3. Low-Pass Ghost Filter: Разрешаем деревьям быть существенно ниже соседей (до 40%).
    """

    def __init__(self, input_path: str, output_img_path: str, progress_callback, error_callback):
        self.input_path = input_path
        self.output_img_path = output_img_path
        self.on_progress = progress_callback
        self.on_error = error_callback

        # --- Параметры ---
        # 1. Очистка
        self.sor_neighbors = 12
        self.sor_std_mul = 2.5  # Мягкая

        # 2. Dual CHM
        self.pixel_size = 0.20  # Высокая детализация
        self.smooth_sigma = 0.8  # Для границ (Watershed)
        self.detection_sigma = 0.35  # Для пиков (очень резко!)

        self.min_tree_dist = 1.1  # Разрешаем очень близкое соседство
        self.min_tree_height = 2.0

        # 3. Anti-Ghost (Low Pass)
        self.ghost_dist_ratio = 0.75
        self.ghost_height_ratio = 0.40  # Разрешаем деревьям быть в 2.5 раза ниже соседа

        # 4. Verticality Check (Новое!)
        # Дерево должно иметь вертикальный разброс точек > 10% от своей высоты
        self.min_verticality = 0.10

        # 5. Radius
        self.radius_expansion = 1.15

        self.clean_points = None
        self.z_ground = 0.0
        self.chm_smooth = None
        self.chm_sharp = None
        self.labels_grid = None
        self.geo_transform = {}
        self.result_trees = []

    def get_trees(self):
        return self.result_trees

    def full_parse(self):
        try:
            self._rep(0)
            if not self._load(): return

            self._denoise()
            self._generate_dual_chm()  # Генерируем две карты
            self._run_dual_watershed()  # Ищем на одной, режем по другой
            self._extract_adaptive()
            self._remove_conflicts()
            self._render()

            self._rep(100)
        except Exception as e:
            if self.on_error: self.on_error(e)

    def print_report(self):
        if not self.result_trees:
            print("No trees.")
            return

        counts = {}
        hs, rs = [], []
        for t in self.result_trees:
            counts[t.shape_type] = counts.get(t.shape_type, 0) + 1
            hs.append(t.height)
            rs.append(t.radius)

        print(f"--- AdaptiveFusion V17 (Spectral Hunter) ---")
        print(f"Trees: {len(self.result_trees)}")
        print(f"  Types: {counts}")
        if hs:
            print(f"  Avg H: {np.mean(hs):.2f}m | Avg R: {np.mean(rs):.2f}m")

    # --- Pipeline ---

    def _rep(self, v):
        if self.on_progress: self.on_progress(float(v))

    def _load(self):
        if not os.path.exists(self.input_path): return False
        las = laspy.read(self.input_path)
        pts = np.vstack((las.x, las.y, las.z)).T

        if hasattr(las, 'classification'):
            gm = (np.array(las.classification) == 2)
            self.z_ground = np.percentile(las.z[gm], 50) if np.any(gm) else np.min(las.z)
        else:
            self.z_ground = np.min(las.z)

        pts[:, 2] -= self.z_ground
        self.clean_points = pts[pts[:, 2] >= 1.0]
        if len(self.clean_points) == 0: raise ValueError("Empty cloud.")
        self._rep(10)
        return True

    def _denoise(self):
        pts = self.clean_points
        if len(pts) < 100: return
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=self.sor_neighbors, workers=-1)
        mean_d = np.mean(dists, axis=1)
        limit = np.mean(mean_d) + self.sor_std_mul * np.std(mean_d)
        self.clean_points = pts[mean_d < limit]
        self._rep(15)

    def _generate_dual_chm(self):
        pts = self.clean_points
        res = self.pixel_size
        x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
        y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
        cols = int(np.ceil((x_max - x_min) / res)) + 1
        rows = int(np.ceil((y_max - y_min) / res)) + 1
        self.geo_transform = {'x_min': x_min, 'y_max': y_max, 'res': res, 'c': cols, 'r': rows}

        ix = ((pts[:, 0] - x_min) / res).astype(np.int32)
        iy = ((y_max - pts[:, 1]) / res).astype(np.int32)
        valid = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
        ix, iy, z = ix[valid], iy[valid], pts[valid, 2]

        idx = iy * cols + ix
        lex = np.lexsort((z, idx))
        s_idx, s_z = idx[lex], z[lex]
        s_idx_r, s_z_r = s_idx[::-1], s_z[::-1]
        u_idx, u_pos = np.unique(s_idx_r, return_index=True)

        grid = np.zeros((rows, cols), dtype=np.float32)
        grid.flat[u_idx] = s_z_r[u_pos]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)

        # Две версии
        self.chm_smooth = gaussian_filter(grid, sigma=self.smooth_sigma)
        self.chm_sharp = gaussian_filter(grid, sigma=self.detection_sigma)
        self._rep(25)

    def _run_dual_watershed(self):
        # Ищем пики на РЕЗКОЙ карте (видим мелочи)
        min_dist_px = int(self.min_tree_dist / self.pixel_size)
        peaks = peak_local_max(self.chm_sharp, min_distance=min_dist_px, threshold_abs=self.min_tree_height)

        markers = np.zeros_like(self.chm_sharp, dtype=np.int32)
        for i, (r, c) in enumerate(peaks):
            markers[r, c] = i + 1

        # Строим бассейны на ГЛАДКОЙ карте (красивые границы)
        mask = self.chm_smooth > 1.0
        self.labels_grid = watershed(-self.chm_smooth, markers, mask=mask)
        self._rep(40)

    def _extract_adaptive(self):
        self.result_trees = []
        pts = self.clean_points

        ix = ((pts[:, 0] - self.geo_transform['x_min']) / self.pixel_size).astype(np.int32)
        iy = ((self.geo_transform['y_max'] - pts[:, 1]) / self.pixel_size).astype(np.int32)
        valid = (ix >= 0) & (ix < self.geo_transform['c']) & (iy >= 0) & (iy < self.geo_transform['r'])
        ix, iy = ix[valid], iy[valid]
        pts_valid = pts[valid]
        pt_labels = self.labels_grid[iy, ix]
        unique_ids = np.unique(pt_labels)
        unique_ids = unique_ids[unique_ids != 0]

        total = len(unique_ids)
        for k, uid in enumerate(unique_ids):
            mask = (pt_labels == uid)
            c_pts = pts_valid[mask]

            # --- VERTICALITY FILTER (Убийца веток) ---
            # Если точек мало, или они лежат в плоскости (малый std_z) -> это не дерево
            if len(c_pts) < 10: continue

            h_max = np.max(c_pts[:, 2])
            std_z = np.std(c_pts[:, 2])

            # Условие: разброс высот должен быть ощутимым (не плоская ветка)
            # Для маленьких деревьев std_z маленький, но отношение std_z / h_max должно быть > порога
            verticality = std_z / h_max
            if verticality < self.min_verticality:
                continue  # Это скорее всего часть кроны другого дерева

            # Центровка и Радиус (как в V16)
            weights = np.power(c_pts[:, 2], 4)
            cx = np.average(c_pts[:, 0], weights=weights)
            cy = np.average(c_pts[:, 1], weights=weights)

            d_full = np.linalg.norm(c_pts[:, :2] - np.array([cx, cy]), axis=1)
            shape = self._guess_shape(c_pts[:, 2], d_full, h_max)

            if shape == 'conical':
                z_min, z_max = h_max * 0.2, h_max * 0.6
            else:
                z_min, z_max = h_max * 0.5, h_max * 0.9

            mask_slice = (c_pts[:, 2] >= z_min) & (c_pts[:, 2] <= z_max)

            if np.sum(mask_slice) > 5:
                d_slice = d_full[mask_slice]
                raw_rad = np.percentile(d_slice, 95)
            else:
                raw_rad = np.percentile(d_full, 90)

            final_rad = raw_rad * self.radius_expansion
            final_rad = max(1.0, final_rad)
            final_rad = min(final_rad, h_max * 0.6)

            self.result_trees.append(Tree(int(uid), cx, cy, h_max, final_rad, shape))

            if k % 100 == 0: self._rep(40 + (k / total) * 30)

    def _remove_conflicts(self):
        if not self.result_trees: return
        trees = sorted(self.result_trees, key=lambda t: t.height, reverse=True)
        keep = []
        active = np.ones(len(trees), dtype=bool)

        coords = np.array([[t.x, t.y] for t in trees])
        radii = np.array([t.radius for t in trees])
        heights = np.array([t.height for t in trees])

        for i in range(len(trees)):
            if not active[i]: continue
            keep.append(trees[i])

            cand_idx = np.where(active[i + 1:])[0] + (i + 1)
            if len(cand_idx) == 0: continue

            dists = np.linalg.norm(coords[cand_idx] - coords[i], axis=1)
            overlap_dist = (radii[i] + radii[cand_idx]) * self.ghost_dist_ratio

            is_overlap = dists < overlap_dist
            # RELAXED: Удаляем только если кандидат ОЧЕНЬ низкий (40%)
            is_short = heights[cand_idx] < (heights[i] * self.ghost_height_ratio)

            is_concentric = dists < 0.8  # Супер близко - точно дубль

            remove_idx = cand_idx[(is_overlap & is_short) | is_concentric]
            active[remove_idx] = False

        self.result_trees = keep
        self._rep(90)

    def _guess_shape(self, z, r, h_top):
        if len(z) < 10: return "spherical"
        h = h_top - z
        m = (h > 0) & (r > 0)
        h, r = h[m], r[m]
        if len(h) < 5: return "spherical"

        z_norm = z / h_top
        z_center = np.mean(z_norm)

        h_inv = h
        corr = np.corrcoef(r, h_inv)[0, 1]

        if corr > 0.70 or z_center < 0.45:
            return "conical"
        return "spherical"

    def _render(self):
        # Рисуем Sharp версию, чтобы видеть детали
        if self.chm_sharp is None: return
        chm_norm = self.chm_sharp.copy()
        vmin, vmax = np.percentile(chm_norm[chm_norm > 0], 5), np.percentile(chm_norm, 99)
        chm_norm = np.clip(chm_norm, vmin, vmax)
        chm_norm = (chm_norm - vmin) / (vmax - vmin + 1e-6)

        h, w = chm_norm.shape
        rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
        rgb_img[..., 1] = (chm_norm * 200 + 55).astype(np.uint8)
        rgb_img[..., 0] = (np.clip(chm_norm - 0.5, 0, 0.5) * 2 * 255).astype(np.uint8)
        rgb_img[..., 2] = ((1 - chm_norm) * 30).astype(np.uint8)
        rgb_img[self.chm_sharp <= 0.1] = [20, 20, 25]

        base_img = Image.fromarray(rgb_img)
        scale = 2.0
        w_new, h_new = int(w * scale), int(h * scale)
        base_img = base_img.resize((w_new, h_new), resample=Image.BILINEAR)

        overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        x_min = self.geo_transform['x_min']
        y_max = self.geo_transform['y_max']
        gsd = self.geo_transform['res'] / scale

        for t in self.result_trees:
            cx = (t.x - x_min) / gsd
            cy = (y_max - t.y) / gsd
            r_px = t.radius / gsd

            c_s = (255, 80, 50, 220) if t.shape_type == 'conical' else (255, 220, 0, 200)
            c_f = (255, 80, 50, 40) if t.shape_type == 'conical' else (255, 220, 0, 40)

            draw.ellipse([cx - r_px, cy - r_px, cx + r_px, cy + r_px], fill=c_f, outline=c_s, width=int(2 * scale))
            draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(255, 255, 255, 255))

        base_img = base_img.convert("RGBA")
        final_img = Image.alpha_composite(base_img, overlay)
        final_img = final_img.resize((w, h), resample=Image.LANCZOS)
        final_img = final_img.convert("RGB")
        final_img.save(self.output_img_path)