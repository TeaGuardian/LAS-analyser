import math
import numpy as np
from .base import Tree
from scipy.optimize import linear_sum_assignment


class TreeComparator:
    """
    Класс для сравнения результатов парсинга с эталоном.
    Использует решение задачи о назначении (Hungarian algorithm) для поиска
    оптимальных пар деревьев и рассчитывает метрику на основе RMSE.
    """

    def __init__(self, reference_trees: list, predicted_trees: list):
        """
        :param reference_trees: Список эталонных деревьев (Ground Truth).
        :param predicted_trees: Список найденных алгоритмом деревьев.
        """
        self.ref = reference_trees
        self.pred = predicted_trees
        self.matches = []  # Список кортежей (ref_idx, pred_idx, similarity)
        self.unmatched_ref = []  # Индексы ненайденных деревьев
        self.unmatched_pred = []  # Индексы лишних деревьев (шума)

        # Статистика
        self.tp = 0  # True Positives (нашли и похоже)
        self.fp = 0  # False Positives (нашли лишнее)
        self.fn = 0  # False Negatives (не нашли)
        self.avg_sim = 0.0  # Средняя схожесть совпавших пар

        # Выполняем сопоставление сразу при инициализации
        self._match_trees()

    def calculate_score(self) -> float:
        """
        Возвращает оценку от 0.0 до 100.0.
        Оценка базируется на (1 - RMSE).
        """
        n_ref = len(self.ref)
        n_pred = len(self.pred)

        # Граничные случаи
        if n_ref == 0 and n_pred == 0:
            return 100.0
        if n_ref == 0 or n_pred == 0:
            return 0.0

        # 1. Сумма квадратных ошибок для совпавших пар
        # Error = (1 - similarity). Если sim=1, err=0. Если sim=0.8, err=0.2.
        squared_error_sum = sum([(1.0 - sim) ** 2 for _, _, sim in self.matches])

        # 2. Штрафы за пропуски и лишние
        # Каждое пропущенное или лишнее дерево - это ошибка 1.0 (квадрат тоже 1.0)
        squared_error_sum += len(self.unmatched_ref) * 1.0
        squared_error_sum += len(self.unmatched_pred) * 1.0

        # 3. Нормализация
        # Делим на макс. количество объектов, которое могло быть
        # Это делает метрику жесткой, но справедливой
        normalization_factor = max(n_ref, n_pred)

        mse = squared_error_sum / normalization_factor
        rmse = math.sqrt(mse)

        # Инвертируем: 0 ошибки -> 100 баллов
        score = 100.0 * (1.0 - rmse)

        return max(0.0, score)

    def print_stats(self):
        """Выводит подробную статистику сравнения."""
        score = self.calculate_score()
        print(f"--- Результаты Сравнения (Score: {score:.1f}/100) ---")
        print(f"Эталон: {len(self.ref)} шт. | Найдено: {len(self.pred)} шт.")
        print(f"Совпадений (TP): {self.tp} (ср. схожесть: {self.avg_sim:.2f})")
        print(f"Пропущено (FN):  {self.fn}")
        print(f"Лишних (FP):     {self.fp}")

        if self.tp > 0:
            # Пример самого плохого совпадения
            worst_match = min(self.matches, key=lambda x: x[2])
            r = self.ref[worst_match[0]]
            p = self.pred[worst_match[1]]
            print(f"Худшее совпадение ({worst_match[2]:.2f}):")
            print(f"  Ref: {r}")
            print(f"  Pred: {p}")

    def _match_trees(self):
        """
        Решает задачу линейного назначения (Linear Assignment Problem).
        Строит матрицу "стоимости" (1 - схожесть) и ищет минимум.
        """
        n_r = len(self.ref)
        n_p = len(self.pred)

        if n_r == 0 or n_p == 0:
            self.unmatched_ref = list(range(n_r))
            self.unmatched_pred = list(range(n_p))
            self.fn = n_r
            self.fp = n_p
            return

        # 1. Строим матрицу стоимости (Cost Matrix)
        # Строки - Reference, Столбцы - Predicted
        # Значение = 1.0 - similarity (потому что алгоритм ищет минимум издержек)
        cost_matrix = np.zeros((n_r, n_p))

        for i in range(n_r):
            for j in range(n_p):
                # Используем перегруженный оператор == класса Tree
                sim = self.ref[i] == self.pred[j]
                cost_matrix[i, j] = 1.0 - sim

        # 2. Венгерский алгоритм
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 3. Разбор результатов
        # row_ind, col_ind - это индексы оптимальных пар

        # Порог "мусорности". Если лучшее совпадение имеет сходство < 0.1,
        # считаем, что это не пара, а пропуск + ложное срабатывание.
        MATCH_THRESHOLD = 0.15

        matched_pred_indices = set()
        matched_ref_indices = set()

        total_sim = 0.0

        for r, p in zip(row_ind, col_ind):
            similarity = 1.0 - cost_matrix[r, p]

            if similarity >= MATCH_THRESHOLD:
                self.matches.append((r, p, similarity))
                matched_ref_indices.add(r)
                matched_pred_indices.add(p)
                total_sim += similarity
            else:
                # Даже оптимальная пара слишком плоха
                pass

        # Заполняем списки пропусков
        self.unmatched_ref = [i for i in range(n_r) if i not in matched_ref_indices]
        self.unmatched_pred = [j for j in range(n_p) if j not in matched_pred_indices]

        # Считаем метрики
        self.tp = len(self.matches)
        self.fn = len(self.unmatched_ref)
        self.fp = len(self.unmatched_pred)
        self.avg_sim = total_sim / self.tp if self.tp > 0 else 0.0


class TreeStatsComparator:
    """
    Расширенный компаратор, который оценивает качество совпадения
    отдельно по 4 параметрам: Тип, Высота, Радиус, Позиция.
    """

    def __init__(self, reference_trees: list, predicted_trees: list):
        """
        :param reference_trees: Список эталонных деревьев.
        :param predicted_trees: Список найденных алгоритмом деревьев.
        """
        self.ref = reference_trees
        self.pred = predicted_trees

        # Список словарей с детальной инфой по совпадениям:
        # [{'ref_idx': int, 'pred_idx': int, 'type': float, 'height': float, 'radius': float, 'pos': float}, ...]
        self.match_details = []

        self.unmatched_ref_count = 0
        self.unmatched_pred_count = 0

        # Статистика совпадений
        self.tp = 0

        # Выполняем сопоставление
        self._match_trees()

    def calculate_score(self) -> dict:
        """
        Возвращает словарь с оценками (0..100) по каждой категории.
        Ключи: 'total', 'type', 'height', 'radius', 'position'.
        """
        n_ref = len(self.ref)
        n_pred = len(self.pred)
        norm_factor = max(n_ref, n_pred)

        if norm_factor == 0:
            return {k: 100.0 for k in ['total', 'type', 'height', 'radius', 'position']}

        # Количество ошибок (пропуски + лишние)
        # Каждое пропущенное/лишнее дерево дает ошибку 1.0 (полное несовпадение) для всех категорий
        miss_errors = (self.unmatched_ref_count + self.unmatched_pred_count) * 1.0

        scores = {}
        # Категории, возвращаемые compare_with
        categories = ['type', 'height', 'radius', 'pos']

        # Считаем RMSE для каждой категории отдельно
        sum_sq_err_total = 0.0

        for cat in categories:
            # Сумма квадратов ошибок совпавших пар: (1 - sim)^2
            match_sq_err = sum([(1.0 - m[cat]) ** 2 for m in self.match_details])

            # Полная ошибка
            total_sq_err = match_sq_err + miss_errors

            # RMSE
            rmse = math.sqrt(total_sq_err / norm_factor)

            # Score
            scores[cat] = max(0.0, 100.0 * (1.0 - rmse))

            # Для общего итога накапливаем ошибки (как среднее арифметическое ошибок категорий)
            # Или можно усреднить итоговые скоры. Сделаем усреднение скоров ниже.

        # Итоговая оценка как среднее по 4 параметрам
        # (или можно посчитать RMSE от средней схожести, математика будет чуть другой)
        scores['total'] = (scores['type'] + scores['height'] + scores['radius'] + scores['pos']) / 4.0

        # Переименуем 'pos' в 'position' для красоты
        scores['position'] = scores.pop('pos')

        return scores

    def print_stats(self):
        """Выводит детальную разбивку по метрикам."""
        scores = self.calculate_score()

        print(f"--- Детальный анализ результатов ---")
        print(f"Эталон: {len(self.ref)} | Найдено: {len(self.pred)}")
        print(
            f"Совпало (TP): {self.tp} | Пропущено (FN): {self.unmatched_ref_count} | Лишние (FP): {self.unmatched_pred_count}")
        print("-" * 40)
        print(f"ОБЩАЯ ОЦЕНКА:    {scores['total']:.1f} / 100")
        print("-" * 40)
        print(f"  По типу кроны: {scores['type']:.1f}")
        print(f"  По высоте:     {scores['height']:.1f}")
        print(f"  По радиусу:    {scores['radius']:.1f}")
        print(f"  По координатам:{scores['position']:.1f}")
        print("=" * 40)

    def _match_trees(self):
        """
        Строит матрицу стоимости на основе среднего арифметического 4-х параметров
        и ищет оптимальные пары.
        """
        n_r = len(self.ref)
        n_p = len(self.pred)

        if n_r == 0 or n_p == 0:
            self.unmatched_ref_count = n_r
            self.unmatched_pred_count = n_p
            return

        # 1. Строим матрицу стоимости
        cost_matrix = np.zeros((n_r, n_p))

        # Кэш схожестей, чтобы не пересчитывать compare_with дважды
        # cache[i][j] = (s_type, s_h, s_r, s_p)
        sim_cache = {}

        for i in range(n_r):
            for j in range(n_p):
                # Получаем 4 компонента схожести
                # sim_type, sim_height, sim_radius, sim_pos
                sims = self.ref[i].compare_with(self.pred[j])
                sim_cache[(i, j)] = sims

                # Общая схожесть для матчинга - среднее арифметическое
                avg_sim = sum(sims) / 4.0
                cost_matrix[i, j] = 1.0 - avg_sim

        # 2. Венгерский алгоритм
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 3. Фильтрация и сохранение результатов
        # Порог средней схожести, ниже которого считаем, что это не пара
        MATCH_THRESHOLD = 0.15

        matched_ref_indices = set()
        matched_pred_indices = set()

        for r, p in zip(row_ind, col_ind):
            # Достаем сохраненные значения
            s_t, s_h, s_r, s_p = sim_cache[(r, p)]
            avg_sim = (s_t + s_h + s_r + s_p) / 4.0

            if avg_sim >= MATCH_THRESHOLD:
                self.match_details.append({
                    'ref_idx': r,
                    'pred_idx': p,
                    'type': s_t,
                    'height': s_h,
                    'radius': s_r,
                    'pos': s_p
                })
                matched_ref_indices.add(r)
                matched_pred_indices.add(p)
                self.tp += 1
            else:
                # Пара найдена алгоритмом, но она слишком плохая (слишком далеко или совсем не то)
                pass

        self.unmatched_ref_count = n_r - len(matched_ref_indices)
        self.unmatched_pred_count = n_p - len(matched_pred_indices)