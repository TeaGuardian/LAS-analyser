import os
import laspy
import numpy as np
from .base import Tree


class LasGenerator:
    """
    Генератор синтетических LAS-файлов (v2.1 Realistic + Transparency).
    Генерирует лес с физически правдоподобной структурой:
    - Плотный ствол (Trunk).
    - Переменная плотность кроны.
    - ASPRS классификация.
    - Симуляция плотности/прозрачности леса (LiDAR penetration).
    """

    def __init__(self,
                 output_path: str,
                 point_step: float = 0.1,
                 area_size: tuple = (50, 50, 45),
                 noise_level: float = 0.1,
                 dropout_rate: float = 0.1,
                 overlap_factor: float = 0.2,
                 mixing_factor: float = 0.1,
                 gap_factor: float = 0.0,
                 transparency: float = 0.8,
                 seed: int = 42,
                 progress_callback=None,
                 error_callback=None):

        self.output_path = output_path
        self.step = point_step
        self.dims = area_size

        self.noise = np.clip(noise_level, 0, 1)
        self.dropout = np.clip(dropout_rate, 0, 1)
        self.overlap = np.clip(overlap_factor, 0, 1)
        self.mixing = np.clip(mixing_factor, 0, 1)
        self.gap_factor = np.clip(gap_factor, 0, 1)
        self.transparency = np.clip(transparency, 0.0, 1.0)  # 0=Solid, 1=Invisible

        self.seed = seed
        self.on_progress = progress_callback
        self.on_error = error_callback

        self.rng = np.random.default_rng(seed)

        self.generated_trees = []
        self.all_points = []
        self.river_params = None
        self.lakes = []

    def get_trees(self):
        return self.generated_trees

    def generate(self):
        try:
            self._report(0)
            self._setup_voids()

            # 1. Земля
            self._generate_ground()
            self._report(15)

            # 2. План леса
            tree_params = self._plan_forest()
            self._report(25)

            # 3. Генерация деревьев (Ствол + Крона)
            total = len(tree_params)
            if total > 0:
                for i, p in enumerate(tree_params):
                    self._create_tree_structure(p)
                    if i % 10 == 0: self._report(25 + (i / total) * 55)

            # 4. Шум
            self._generate_noise()
            self._report(85)

            # 5. Симуляция прозрачности/окклюзии (НОВЫЙ ШАГ)
            self._apply_transparency()
            self._report(95)

            # 6. Сохранение
            self._save()
            self._report(100)

        except Exception as e:
            if self.on_error:
                self.on_error(e)
            else:
                print(f"Gen Error: {e}")

    def print_report(self):
        if not self.generated_trees:
            print("No data.")
            return

        counts = {}
        hs, rs = [], []
        for t in self.generated_trees:
            counts[t.shape_type] = counts.get(t.shape_type, 0) + 1
            hs.append(t.height)
            rs.append(t.radius)

        # Подсчет реального количества точек
        total_pts = sum([len(chunk) for chunk in self.all_points])

        print(f"--- Realistic Generator Report (v2.1) ---")
        print(f"File: {os.path.basename(self.output_path)}")
        print(f"Trees: {len(self.generated_trees)} | Overlap: {self.overlap * 100:.0f}%")
        print(f"Transparency: {self.transparency:.2f} | Total Points: {total_pts}")
        print(f"Types: {counts}")
        if hs:
            print(f"Avg H: {np.mean(hs):.2f}m | Avg R: {np.mean(rs):.2f}m")

    # --- Logic ---

    def _report(self, v):
        if self.on_progress: self.on_progress(float(v))

    def _setup_voids(self):
        if self.gap_factor <= 0.01: return
        if self.gap_factor > 0.2:
            self.river_params = {
                'axis': 'x',
                'amp': self.rng.uniform(2, self.dims[1] / 4),
                'freq': self.rng.uniform(0.05, 0.15),
                'phase': self.rng.uniform(0, np.pi),
                'center': self.rng.uniform(self.dims[1] * 0.3, self.dims[1] * 0.7),
                'width': 2.5 + (self.dims[1] * 0.2 * self.gap_factor)
            }
        n_lakes = int(self.gap_factor * 4)
        for _ in range(n_lakes):
            self.lakes.append({
                'x': self.rng.uniform(0, self.dims[0]),
                'y': self.rng.uniform(0, self.dims[1]),
                'r': self.rng.uniform(2.0, self.dims[0] * 0.1)
            })

    def _is_void(self, x, y):
        if self.gap_factor <= 0.01: return False
        if self.river_params:
            p = self.river_params
            ry = p['amp'] * np.sin(p['freq'] * x + p['phase']) + p['center']
            if abs(y - ry) < (p['width'] / 2): return True
        for l in self.lakes:
            if np.hypot(x - l['x'], y - l['y']) < l['r']: return True
        return False

    def _generate_ground(self):
        xr = np.arange(0, self.dims[0], self.step)
        yr = np.arange(0, self.dims[1], self.step)
        xx, yy = np.meshgrid(xr, yr)
        zz = 1.5 * np.sin(0.05 * xx) + 1.5 * np.cos(0.05 * yy)
        zz -= np.min(zz)
        n = xx.size
        xf = xx.flatten() + self.rng.uniform(-0.05, 0.05, n)
        yf = yy.flatten() + self.rng.uniform(-0.05, 0.05, n)
        zf = zz.flatten() + self.rng.uniform(-0.02, 0.02, n)
        cf = np.full(n, 2)
        self.all_points.append(np.stack((xf, yf, zf, cf), axis=1))

    def _plan_forest(self):
        trees = []
        density_factor = 0.5 + self.overlap
        count = int((self.dims[0] * self.dims[1] * density_factor) / 25.0)

        for _ in range(count * 5):
            if len(trees) >= count: break
            h = self.rng.triangular(4, 25, 35)
            r = 1.5 + (h / 35.0) * 3.5
            x = self.rng.uniform(0, self.dims[0])
            y = self.rng.uniform(0, self.dims[1])
            if self._is_void(x, y): continue

            valid = True
            for t in trees:
                d = np.hypot(x - t['x'], y - t['y'])
                limit = (r + t['r']) * (1.0 - self.overlap)
                limit = max(0.5, limit)
                if d < limit:
                    valid = False
                    break

            if valid:
                mid = self.dims[0] / 2
                base = 'conical' if x < mid else 'spherical'
                if self.rng.random() < self.mixing:
                    base = 'spherical' if base == 'conical' else 'conical'
                t_dat = {'id': len(trees), 'x': x, 'y': y, 'h': h, 'r': r, 'type': base}
                trees.append(t_dat)
                self.generated_trees.append(Tree(t_dat['id'], x, y, h, r, base))
        return trees

    def _create_tree_structure(self, p):
        h, r = p['h'], p['r']
        cx, cy = p['x'], p['y']
        z_base_terrain = 1.5 * np.sin(0.05 * cx) + 1.5 * np.cos(0.05 * cy)
        Z_offset = z_base_terrain + 3.0

        pts_list = []

        # Ствол
        n_trunk = int(h * 100)
        tz = self.rng.uniform(0, h * 0.9, n_trunk)
        tr = self.rng.uniform(0, 0.15, n_trunk)
        ta = self.rng.uniform(0, 2 * np.pi, n_trunk)
        tx = cx + tr * np.cos(ta)
        ty = cy + tr * np.sin(ta)
        pts_list.append(np.stack((tx, ty, tz + Z_offset), axis=1))

        # Крона
        n_foliage = int((r ** 2 * h) * 40)
        n_foliage = min(n_foliage, 50000)
        fr_raw = self.rng.random(n_foliage)
        fr = r * fr_raw
        fa = self.rng.uniform(0, 2 * np.pi, n_foliage)
        fz = self.rng.uniform(0, h, n_foliage)
        fx = cx + fr * np.cos(fa)
        fy = cy + fr * np.sin(fa)

        d_xy = fr
        if p['type'] == 'conical':
            valid = d_xy <= (r * (1 - fz / h)*2.6)
        else:
            crown_start = h * 0.2
            crown_h = h * 0.8
            center_z = crown_start + crown_h / 2
            term_z = ((fz - center_z) / (crown_h / 2)) ** 2
            term_xy = (d_xy / r) ** 2
            valid = (term_z + term_xy) <= 1.0
            valid |= (d_xy < r * 0.3) & (fz < crown_start)

        fx = fx[valid]
        fy = fy[valid]
        fz = fz[valid]
        pts_list.append(np.stack((fx, fy, fz + Z_offset), axis=1))

        full_pts = np.concatenate(pts_list, axis=0)
        rel_z = full_pts[:, 2] - Z_offset
        classes = np.zeros(len(full_pts), dtype=np.uint8)
        classes[rel_z < 0.5] = 2
        classes[(rel_z >= 0.5) & (rel_z < 2.0)] = 3
        classes[(rel_z >= 2.0) & (rel_z < 5.0)] = 4
        classes[rel_z >= 5.0] = 5

        if self.dropout > 0:
            keep = self.rng.random(len(full_pts)) > self.dropout
            full_pts = full_pts[keep]
            classes = classes[keep]

        self.all_points.append(np.column_stack((full_pts, classes)))

    def _generate_noise(self):
        if self.noise <= 0: return
        cnt = sum(len(x) for x in self.all_points)
        n = int(cnt * self.noise)
        if n == 0: return
        nx = self.rng.uniform(0, self.dims[0], n)
        ny = self.rng.uniform(0, self.dims[1], n)
        nz = self.rng.uniform(0, self.dims[2], n)
        nc = self.rng.choice([3, 4, 5], n)
        self.all_points.append(np.stack((nx, ny, nz, nc), axis=1))

    def _apply_transparency(self):
        """
        Фильтр видимости. Удаляет точки, которые "заслонены" верхними точками
        в зависимости от коэффициента self.transparency.
        """
        # Если 1.0 - полная видимость, ничего делать не надо
        if self.transparency >= 0.99:
            return

        if not self.all_points:
            return

        # Собираем всё в один массив
        full_cloud = np.concatenate(self.all_points, axis=0)

        # Параметры сетки (разрешение симуляции окклюзии)
        # Чем меньше, тем детальнее "тень"
        grid_res = max(self.step, 0.25)

        x = full_cloud[:, 0]
        y = full_cloud[:, 1]
        z = full_cloud[:, 2]

        # 1. Дискретизация пространства (Grid index)
        idx_x = (x / grid_res).astype(np.int32)
        idx_y = (y / grid_res).astype(np.int32)

        # 2. Поиск самой высокой точки в каждой ячейке (Max Z buffer)
        # Создаем уникальный 1D индекс для ячейки
        width_approx = int(np.max(idx_x)) + 2
        flat_idx = idx_y * width_approx + idx_x

        # Сортируем точки: сначала по ячейке, потом по Z (сверху вниз)
        # -z означает, что sort будет по убыванию
        sort_order = np.lexsort((-z, flat_idx))
        sorted_pts = full_cloud[sort_order]
        sorted_flat = flat_idx[sort_order]
        sorted_z = sorted_pts[:, 2]

        # Находим начало каждой новой ячейки в отсортированном массиве
        # unique_indices указывает на ПЕРВЫЙ элемент группы, который благодаря сортировке
        # является самым высоким (Max Z)
        _, first_indices = np.unique(sorted_flat, return_index=True)

        # Создаем массив MaxZ такой же длины, как sorted_pts
        # Сначала находим индексы повторений (inverse)
        _, inv_indices = np.unique(sorted_flat, return_inverse=True)
        # Значения Max Z для каждой группы
        max_z_values = sorted_z[first_indices]
        # Растягиваем обратно на все точки
        z_buffers = max_z_values[inv_indices]

        # 3. Расчет вероятности выживания точки
        # delta_z: глубина точки относительно поверхности
        delta_z = z_buffers - sorted_z

        # Формула вероятности: P = transparency ^ delta_z
        # Если T=0: 0^0 = 1 (верхняя точка), 0^N = 0 (остальные удаляются)
        # Если T=0.5, delta=1m: P=0.5. delta=2m: P=0.25
        # Добавляем epsilon к delta_z для нижних точек, чтобы избежать float error
        # Но для T=0 это работает идеально благодаря поведению np.power

        probs = np.power(self.transparency, delta_z)

        # Генерируем случайные числа для теста
        random_rolls = self.rng.random(len(probs))
        mask_keep = random_rolls <= probs

        # Применяем фильтр к отсортированному массиву
        filtered_cloud = sorted_pts[mask_keep]

        # Перезаписываем список точек
        self.all_points = [filtered_cloud]

    def _save(self):
        if not self.all_points: return
        arr = np.concatenate(self.all_points, axis=0)
        mins = np.min(arr[:, :3], axis=0)
        hdr = laspy.LasHeader(point_format=3, version="1.2")
        hdr.scales = [0.001, 0.001, 0.001]
        hdr.offsets = mins
        las = laspy.LasData(hdr)
        las.x = arr[:, 0]
        las.y = arr[:, 1]
        las.z = arr[:, 2]
        las.classification = arr[:, 3].astype(np.uint8)
        las.write(self.output_path)
        self.all_points = []