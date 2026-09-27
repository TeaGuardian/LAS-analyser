import math


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
