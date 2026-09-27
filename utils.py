import sys
import os
import time
import shutil
import logging
import uuid
import threading
import itertools
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Callable, Any
import backend  # Ваша библиотека

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ForestBenchmark")

# КОНСТАНТЫ
MIN_FREE_SPACE_BYTES = 500 * 1024 * 1024
MAX_REGEN_ATTEMPTS = 5
WATCHDOG_TIMEOUT = 120  # 5 минут (300 секунд) до признания задачи зависшей


@dataclass
class TaskParams:
    noise_level: float
    dropout_rate: float
    overlap_factor: float
    mixing_factor: float
    gap_factor: float
    seed: int = 42
    generation_attempt: int = 0

    def get_filename_base(self) -> str:
        return (f"N{self.noise_level:.2f}_D{self.dropout_rate:.2f}_"
                f"O{self.overlap_factor:.2f}_M{self.mixing_factor:.2f}_"
                f"G{self.gap_factor:.2f}")


@dataclass
class FileContext:
    file_id: str
    params: TaskParams
    las_path: str
    ref_trees: List[Any]
    parsers_to_run: List[str]
    remaining_reads: int
    is_invalid: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class BenchmarkManager:
    def __init__(self,
                 las_path: str = "storage/las",
                 img_path: str = "storage/img",
                 gen_workers: int = 2,
                 parse_workers: int = 4,
                 compare_workers: int = 2,
                 area_size: tuple = (100, 100, 100)):

        self.las_path = Path(las_path)
        self.img_path = Path(img_path)
        self.area_size = area_size

        self.las_path.mkdir(parents=True, exist_ok=True)
        self.img_path.mkdir(parents=True, exist_ok=True)

        self.gen_pool = ThreadPoolExecutor(max_workers=gen_workers, thread_name_prefix="Gen")
        self.parse_pool = ThreadPoolExecutor(max_workers=parse_workers, thread_name_prefix="Parse")
        self.compare_pool = ThreadPoolExecutor(max_workers=compare_workers, thread_name_prefix="Comp")

        self.results: List[Dict] = []
        self.results_lock = threading.Lock()

        self._is_running = False
        self.total_ops = 0
        self.completed_ops = 0
        self.ops_lock = threading.Lock()

        # --- WATCHDOG REGISTRY ---
        # task_uid -> {'time': last_update_ts, 'type': 'gen'|'parse', 'args': ...}
        self._watchdog_registry = {}
        self._watchdog_lock = threading.Lock()
        self._watchdog_thread = None

        # Callbacks
        self.on_global_progress: Callable[[int, int], None] = lambda c, t: None
        self.on_task_status: Callable[[str, str, int, bool], None] = lambda uid, name, p, f: None
        self.on_finished: Callable[[List[Dict]], None] = lambda r: None
        self.on_result: Callable[[Dict], None] = lambda r: None

    def _float_range(self, start, stop, step):
        val = start
        while val <= stop + 1e-9:
            yield round(val, 2)
            val += step
    """
    def generate_tasks(self, step: float) -> List[TaskParams]:
        noises = list(self._float_range(0.0, 1, step))
        dropouts = list(self._float_range(0.0, 1, step))
        overlaps = list(self._float_range(0.0, 1, step))
        mixings = list(self._float_range(0.0, 1, step))
        gaps = list(self._float_range(0.0, 1, step))
        combinations = itertools.product(noises, dropouts, overlaps, mixings, gaps)
        return [TaskParams(n, d, o, m, g) for n, d, o, m, g in combinations]
        """

    def generate_tasks(self, step: float) -> List[TaskParams]:
        """
        Генерирует задачи по принципу 'Один за раз':
        Меняем один параметр от 0.0 до 1, пока остальные остаются 0.0.
        """
        values = list(self._float_range(0, 1, step))
        unique_params = set()
        unique_params.add((0.0, 0.0, 0.0, 0.0, 0.0))
        for val in values:
            unique_params.add((val, 0.0, 0.0, 0.0, 0.0))
        for val in values:
            unique_params.add((0.0, val, 0.0, 0.0, 0.0))
        for val in values:
            unique_params.add((0.0, 0.0, val, 0.0, 0.0))
        for val in values:
            unique_params.add((0.0, 0.0, 0.0, val, 0.0))
        for val in values:
            unique_params.add((0.0, 0.0, 0.0, 0.0, val))
        sorted_params = sorted(list(unique_params))

        tasks = []
        for n, d, o, m, g in sorted_params:
            tasks.append(TaskParams(
                noise_level=n,
                dropout_rate=d,
                overlap_factor=o,
                mixing_factor=m,
                gap_factor=g
            ))

        return tasks

    def start(self, step: float = 0.1):
        if self._is_running: return
        self.results = []
        self.completed_ops = 0
        tasks = self.generate_tasks(step)
        num_parsers = len(backend.PARSERS)
        self.total_ops = len(tasks) * (1 + num_parsers * 2)

        self._is_running = True

        # Запуск Watchdog
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

        logger.info(f"Starting benchmark: {len(tasks)} files on {self.las_path}")

        for params in tasks:
            if not self._is_running: break
            self.gen_pool.submit(self._job_generate, params)

    def stop(self):
        self._is_running = False
        self.gen_pool.shutdown(wait=False)
        self.parse_pool.shutdown(wait=False)
        self.compare_pool.shutdown(wait=False)

    def _update_global_progress(self):
        with self.ops_lock:
            self.completed_ops += 1
            c = self.completed_ops
            t = self.total_ops
        self.on_global_progress(c, t)
        if c >= t:
            self._is_running = False
            self.on_finished(self.results)

    def _check_disk_space(self) -> bool:
        try:
            usage = shutil.disk_usage(self.las_path)
            return usage.free > MIN_FREE_SPACE_BYTES
        except Exception:
            return True

    def _wait_for_disk(self, job_id):
        while self._is_running and not self._check_disk_space():
            self.on_task_status(job_id, "Low Disk Space (Waiting)", 0, False)
            time.sleep(2)

    # --- WATCHDOG METHODS ---
    def _register_monitor(self, uid, type, args):
        with self._watchdog_lock:
            self._watchdog_registry[uid] = {
                'time': time.time(),
                'type': type,
                'args': args,
                'zombie': False
            }

    def _update_monitor(self, uid):
        with self._watchdog_lock:
            if uid in self._watchdog_registry:
                if self._watchdog_registry[uid]['zombie']:
                    return False  # Задача помечена как зомби, игнорируем её прогресс
                self._watchdog_registry[uid]['time'] = time.time()
                return True
        return False

    def _unregister_monitor(self, uid):
        with self._watchdog_lock:
            if uid in self._watchdog_registry:
                del self._watchdog_registry[uid]

    def _watchdog_loop(self):
        """Фоновый поток для проверки зависших задач."""
        while self._is_running:
            time.sleep(10)  # Проверка каждые 10 сек
            now = time.time()
            to_restart = []

            with self._watchdog_lock:
                for uid, data in self._watchdog_registry.items():
                    if data['zombie']: continue  # Уже перезапущена

                    if now - data['time'] > WATCHDOG_TIMEOUT:
                        logger.warning(f"Task {uid} hung (no response for {WATCHDOG_TIMEOUT}s). Restarting.")
                        data['zombie'] = True  # Помечаем старую как зомби
                        to_restart.append(data)

            for task_data in to_restart:
                # Перезапуск логики
                if task_data['type'] == 'gen':
                    # args = (params,)
                    self.on_task_status(str(uuid.uuid4()), "Restarting Hung Gen", 0, True)
                    self.gen_pool.submit(self._job_generate, *task_data['args'])

                elif task_data['type'] == 'parse':
                    # args = (ctx, parser_name)
                    self.on_task_status(str(uuid.uuid4()), "Restarting Hung Parse", 0, True)
                    self.parse_pool.submit(self._job_parse, *task_data['args'])

    # ==========================================
    # ЭТАП 1: ГЕНЕРАЦИЯ (с Watchdog)
    # ==========================================
    def _job_generate(self, params: TaskParams):
        # Уникальный ID для текущего запуска (потока)
        job_id = str(uuid.uuid4())

        # Регистрируем в Watchdog
        self._register_monitor(job_id, 'gen', (params,))

        try:
            if not self._is_running: return

            if params.generation_attempt > MAX_REGEN_ATTEMPTS:
                logger.error(f"Max attempts reached for {params.get_filename_base()}")
                skipped_ops = 1 + len(backend.PARSERS) * 2
                with self.ops_lock: self.completed_ops += skipped_ops
                self.on_global_progress(self.completed_ops, self.total_ops)
                return

            base_name = params.get_filename_base()
            las_path = str(self.las_path / f"{base_name}.las")

            success = False
            for attempt in range(3):
                if not self._is_running: return

                self._wait_for_disk(job_id)

                # Проверка на зомби (если нас уже перезапустили)
                if not self._update_monitor(job_id): return

                try:
                    def progress_wrapper(p):
                        # Обновляем Watchdog и UI
                        if self._update_monitor(job_id):
                            self.on_task_status(job_id, f"Gen (Att {params.generation_attempt}): {base_name}", int(p),
                                                False)

                    generator = backend.LasGenerator(
                        output_path=las_path, area_size=self.area_size,
                        noise_level=params.noise_level, dropout_rate=params.dropout_rate,
                        overlap_factor=params.overlap_factor, mixing_factor=params.mixing_factor,
                        gap_factor=params.gap_factor, seed=params.seed,
                        progress_callback=progress_wrapper
                    )
                    generator.generate()
                    ref_trees = generator.get_trees()
                    success = True
                    break
                except OSError as e:
                    if e.errno == 28:
                        self._wait_for_disk(job_id); continue
                    else:
                        time.sleep(1)
                except Exception as e:
                    logger.error(f"Gen fail: {e}");
                    time.sleep(1)

            if not success:
                self.on_task_status(job_id, "Gen Failed", 0, True)
                return

            self.on_task_status(job_id, f"Gen: {base_name}", 100, True)
            self._update_global_progress()

            parsers_list = list(backend.PARSERS.keys())
            ctx = FileContext(
                file_id=job_id, params=params, las_path=las_path,
                ref_trees=ref_trees, parsers_to_run=parsers_list,
                remaining_reads=len(parsers_list)
            )
            self._schedule_next_parser(ctx)

        finally:
            self._unregister_monitor(job_id)

    # ==========================================
    # ЭТАП 2: ПАРСИНГ (с Watchdog)
    # ==========================================
    def _schedule_next_parser(self, ctx: FileContext):
        if not self._is_running: return
        next_parser_name = None
        with ctx.lock:
            if not ctx.is_invalid and ctx.parsers_to_run:
                next_parser_name = ctx.parsers_to_run.pop(0)
        if next_parser_name:
            self.parse_pool.submit(self._job_parse, ctx, next_parser_name)

    def _job_parse(self, ctx: FileContext, parser_name: str):
        # Уникальный ID для текущего потока
        task_uid = f"{ctx.file_id}_{parser_name}_{uuid.uuid4().hex[:4]}"

        self._register_monitor(task_uid, 'parse', (ctx, parser_name))

        try:
            if not self._is_running: return

            img_path = str(self.img_path / f"{ctx.params.get_filename_base()}_{parser_name.replace(' ', '_')}.png")

            # Проверка целостности
            file_valid = False
            try:
                if os.path.exists(ctx.las_path) and os.path.getsize(ctx.las_path) > 1024:
                    file_valid = True
            except:
                pass

            if not file_valid:
                need_regen = False
                with ctx.lock:
                    if not ctx.is_invalid:
                        ctx.is_invalid = True
                        need_regen = True
                if need_regen:
                    self.on_task_status(task_uid, "Corrupt -> Regen", 0, True)
                    try:
                        os.remove(ctx.las_path)
                    except:
                        pass
                    ctx.params.generation_attempt += 1
                    self.gen_pool.submit(self._job_generate, ctx.params)
                return

            success = False
            predicted_trees = []
            duration = 0.0

            for attempt in range(3):
                if not self._is_running: return
                if not self._update_monitor(task_uid): return

                try:
                    def progress_wrapper(p):
                        if self._update_monitor(task_uid):
                            self.on_task_status(task_uid, f"Parse: {parser_name}", int(p), False)

                    parser_cls = backend.PARSERS[parser_name]
                    parser = parser_cls(
                        input_path=ctx.las_path, output_img_path=img_path,
                        progress_callback=progress_wrapper, error_callback=lambda e: None
                    )

                    t0 = time.time()
                    parser.full_parse()
                    duration = time.time() - t0
                    predicted_trees = parser.get_trees()
                    success = True
                    break
                except Exception as e:
                    err_msg = str(e)
                    if "zero-size array" in err_msg or "Could only read 0" in err_msg:
                        logger.error(f"Data corruption: {e}")
                        need_regen = False
                        with ctx.lock:
                            if not ctx.is_invalid:
                                ctx.is_invalid = True
                                need_regen = True
                        if need_regen:
                            self.on_task_status(task_uid, "Data Error -> Regen", 0, True)
                            try:
                                os.remove(ctx.las_path)
                            except:
                                pass
                            ctx.params.generation_attempt += 1
                            self.gen_pool.submit(self._job_generate, ctx.params)
                        return
                    time.sleep(1)

            if not success:
                self.on_task_status(task_uid, "Parse Failed", 0, True)
                self._cleanup_las(ctx)
                self._schedule_next_parser(ctx)
                return

            self._cleanup_las(ctx)
            self.on_task_status(task_uid, "Done", 100, True)
            self._update_global_progress()

            self.compare_pool.submit(self._job_compare, ctx, parser_name, predicted_trees, img_path, duration)
            self._schedule_next_parser(ctx)

        finally:
            self._unregister_monitor(task_uid)

    def _cleanup_las(self, ctx: FileContext):
        should_delete = False
        with ctx.lock:
            ctx.remaining_reads -= 1
            if ctx.remaining_reads <= 0:
                should_delete = True
        if should_delete and os.path.exists(ctx.las_path):
            try:
                os.remove(ctx.las_path)
            except:
                pass

    def _job_compare(self, ctx: FileContext, parser_name: str, predicted_trees: List[Any], img_path: str,
                     duration: float):
        if not self._is_running: return

        for attempt in range(3):
            try:
                # === ИЗМЕНЕНИЕ: Используем TreeStatsComparator ===
                # Было: comparator = backend.TreeComparator(...)
                # Стало:
                comparator = backend.TreeStatsComparator(ctx.ref_trees, predicted_trees)
                stats = comparator.calculate_score()
                # stats = {'total': 85.5, 'type': 90.0, 'height': 80.0, ...}

                # Берем total для совместимости со старыми графиками
                total_score = stats['total']

                res = {
                    "parser": parser_name,
                    "score": total_score,  # Общий балл (для старых графиков)
                    "stats": stats,  # Детальная статистика (для новых)
                    "duration": duration,
                    "params": ctx.params.__dict__,
                    "las_file": ctx.las_path,
                    "img_file": img_path,
                    "tree_count_ref": len(ctx.ref_trees),
                    "tree_count_pred": len(predicted_trees)
                }

                with self.results_lock:
                    self.results.append(res)
                self.on_result(res)
                self._update_global_progress()
                return  # Success
            except Exception as e:
                logger.error(f"Compare retry {attempt}: {e}")
                time.sleep(0.5)


import sys
import numpy as np
import math
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt, QRectF, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QCheckBox, QButtonGroup,
    QFrame, QScrollArea, QSizePolicy, QLabel, QSpacerItem
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def get_point_color(color_str):
    base_colors = {"#F5812C": "#8C4A00", "#BF0E0E": "#5A0000", "#06BF5C": "#005A2B", "#0C5ABF": "#002C5A"}

    def to_qcolor(c_str):
        if c_str.startswith('#'): return QColor(c_str)
        if c_str.startswith('rgb('): return QColor(*map(int, c_str[4:-1].split(',')))
        return QColor(c_str)

    qcolor = to_qcolor(color_str)
    if not qcolor.isValid(): return "#000000"
    for base_hex, border_hex in base_colors.items():
        base_color = QColor(base_hex)
        if abs(qcolor.red() - base_color.red()) < 25.5 and abs(qcolor.green() - base_color.green()) < 25.5 and abs(
                qcolor.blue() - base_color.blue()) < 25.5:
            return border_hex
    brightness = 0.299 * qcolor.red() + 0.587 * qcolor.green() + 0.114 * qcolor.blue()
    return "#333333" if brightness > 180 else "#000000" if brightness > 120 else "#FFFFFF" if brightness > 60 else "#CCCCCC"


class CustomToolTip(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.container = QWidget()
        self.container.setStyleSheet("background-color: white; padding: 8px; border-radius: 12px; border-color: black;")
        self.main_layout = QVBoxLayout(self.container)
        self.main_layout.setSpacing(8)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.x_label = QLabel()
        self.x_label.setAlignment(Qt.AlignCenter)
        self.data_container = QWidget()
        self.data_layout = QVBoxLayout(self.data_container)
        self.data_layout.setSpacing(4)
        self.data_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.x_label)
        self.main_layout.addWidget(self.data_container)
        layout = QVBoxLayout(self)
        layout.addWidget(self.container)
        layout.setContentsMargins(0, 0, 0, 0)
        self.font = QFont()
        self.set_text_size(10)

    def set_text_size(self, size_px: int):
        self.font.setPixelSize(size_px)
        self.x_label.setFont(self.font)
        for i in range(self.data_layout.count()):
            widget = self.data_layout.itemAt(i).widget()
            if widget: widget.setFont(self.font)

    def set_group_data(self, x_label, x_value, x_val, data_list):
        # Безопасное форматирование
        try:
            x_float = float(x_value)
            x_str = f"{x_float:.4f}".rstrip('0').rstrip('.')
        except (ValueError, TypeError):
            x_str = str(x_value)

        self.x_label.setText(f"{x_label}: <b>{x_str}</b> [{x_val}]")
        while self.data_layout.count():
            item = self.data_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        for legend, y_value, y_val, color in data_list:
            if legend is None: continue
            try:
                y_float = float(y_value)
                y_str = f"{y_float:.4f}".rstrip('0').rstrip('.')
            except (ValueError, TypeError):
                y_str = str(y_value)

            cell = QLabel(f"{legend}: <b>{y_str}</b> [{y_val}]")
            cell.setAlignment(Qt.AlignCenter)
            cell.setStyleSheet(
                f"background-color: white; border: 2px solid {color}; border-radius: 4px; padding: 4px 4px; min-width: 80px;")
            cell.setFont(self.font)
            self.data_layout.addWidget(cell)
        self.adjustSize()
        min_width = max(200, self.x_label.sizeHint().width() + 20)
        for i in range(self.data_layout.count()):
            widget = self.data_layout.itemAt(i).widget()
            if widget: min_width = max(min_width, widget.sizeHint().width() + 20)
        self.container.setMinimumWidth(min_width)
        self.adjustSize()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect_f = QRectF(self.container.rect().adjusted(0, 0, 1, 1))
        path = QPainterPath()
        path.addRoundedRect(rect_f, 6, 6)
        painter.fillPath(path, QColor(0, 0, 0, 20))
        painter.setPen(QColor(200, 200, 200))
        painter.drawRoundedRect(rect_f, 6, 6)


class GraphWidget(QWidget):
    point_clicked_signal = pyqtSignal(str, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)
        self.control_panel = QWidget()
        self.control_layout = QHBoxLayout(self.control_panel)
        self.control_layout.setAlignment(Qt.AlignLeft)
        self.control_layout.setSpacing(8)
        self.control_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.control_panel)
        self.radio_group = QButtonGroup(self)
        self.radio_group.setExclusive(True)
        self.radio_group.buttonToggled.connect(self._handle_radio_toggle)
        self.none_radio = QRadioButton("Нет")
        self.none_radio.setFixedSize(60, 20)
        self.none_radio.setChecked(True)
        self.radio_group.addButton(self.none_radio)
        self.separator2 = QFrame()
        self.separator2.setFrameShape(QFrame.VLine)
        self.separator2.setFrameShadow(QFrame.Sunken)
        self.separator2.setFixedWidth(2)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.graphs_container = QWidget()
        self.graphs_layout = QVBoxLayout(self.graphs_container)
        self.graphs_layout.setAlignment(Qt.AlignTop)
        self.graphs_layout.setSpacing(2)
        self.graphs_layout.setContentsMargins(0, 10, 0, 0)
        self.scroll_area.setWidget(self.graphs_container)
        main_layout.addWidget(self.scroll_area)
        self.graphs = {}
        self.checkbox_container = QWidget()
        self.checkbox_layout = QHBoxLayout(self.checkbox_container)
        self.checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.checkbox_layout.setSpacing(8)
        self.radio_container = QWidget()
        self.radio_layout = QHBoxLayout(self.radio_container)
        self.radio_layout.setContentsMargins(0, 0, 0, 0)
        self.radio_layout.setSpacing(8)
        self.control_layout.addWidget(self.checkbox_container)
        self.control_layout.addWidget(self.none_radio)
        self.control_layout.addWidget(self.separator2)
        self.control_layout.addWidget(self.radio_container)
        self.control_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.checkbox_graphs = []
        self.radio_graphs = []
        self.base_font_size = 10
        self.font_family = 'Montserrat'
        self._update_fonts()
        self.MIN_GRAPH_HEIGHT = 500
        self.MAX_GRAPH_HEIGHT = 800
        self.resize_timer_id = None
        self.tooltip = CustomToolTip()
        self.tooltip.hide()
        self.tooltip_timer = QTimer(self)
        self.tooltip_timer.setSingleShot(True)
        self.tooltip_timer.timeout.connect(
            lambda: self._update_highlight_and_tooltip(self.current_hover, self.last_x, show_tooltip=True))
        self.current_hover = None
        self.last_x = None
        self.graphs_timers_flags = {}

        self.is_selection_mode = False
        self.selection_marker = None

    def _update_fonts(self):
        control_font = QFont(self.font_family, self.base_font_size)
        self.none_radio.setFont(control_font)
        for name, graph in self.graphs.items():
            graph['control'].setFont(control_font)
            self._update_graph_fonts(graph)
            if graph['visible']: self._redraw_graph_by_timer(name)

    def _update_graph_fonts(self, graph):
        ax = graph['ax']
        ax.title.set_fontsize(self.base_font_size + 2)
        ax.xaxis.label.set_fontsize(self.base_font_size)
        ax.yaxis.label.set_fontsize(self.base_font_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_fontsize(self.base_font_size)
        if ax.get_legend():
            for text in ax.get_legend().get_texts(): text.set_fontsize(self.base_font_size)

    def add_graph(self, name: str, x_name: str, y_name: str, x_val: str, y_val: str, ct: bool, colors: list,
                  legends: list, graph_types: list, approximations: list):
        if name in self.graphs: return
        control = QRadioButton(name) if ct else QCheckBox(name)
        if ct:
            self.radio_group.addButton(control)
            self.radio_layout.addWidget(control)
            self.radio_graphs.append(name)
        else:
            control.stateChanged.connect(self._update_visibility)
            self.checkbox_layout.addWidget(control)
            self.checkbox_graphs.append(name)
        control.setFont(QFont(self.font_family, self.base_font_size))
        fig = Figure(figsize=(8, 3), tight_layout=True)
        canvas = FigureCanvas(fig)
        canvas.mpl_connect("motion_notify_event", lambda e, n=name: self.on_mouse_move(e, n))
        canvas.mpl_connect("figure_leave_event", lambda e, n=name: self.on_figure_leave(e, n))
        canvas.mpl_connect("button_press_event", lambda e, n=name: self.on_mouse_click(e, n))
        ax = fig.add_subplot(111)
        ax.set_xlabel(f"{x_name}, {x_val}")
        ax.set_ylabel(f"{y_name}, {y_val}")
        ax.set_title(name)
        ax.grid(True)
        self._update_graph_fonts({'ax': ax, 'fig': fig})
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.setMinimumHeight(self.MIN_GRAPH_HEIGHT)
        canvas.setMaximumHeight(self.MAX_GRAPH_HEIGHT)

        # FIX: Инициализируем данные пустыми списками [], а не None
        series = [{'color': c, 'border_color': get_point_color(c),
                   'legend': legends[i] if i < len(legends) else f"G {i + 1}",
                   'graph_type': graph_types[i] if i < len(graph_types) else 'line',
                   'approximation': approximations[i] if i < len(approximations) else False,
                   'x_data': [], 'y_data': [], 'artists': {}}
                  for i, c in enumerate(colors)]

        self.graphs[name] = {'control': control, 'canvas': canvas, 'ax': ax, 'fig': fig,
                             'ct': ct, 'x_name': x_name, 'y_name': y_name, 'x_val': x_val,
                             'y_val': y_val, 'visible': False, 'series': series}
        self.graphs_layout.addWidget(canvas)
        canvas.setVisible(False)
        self.update_graphs_size()

    def set_data(self, name: str, x_data_list: list, y_data_list: list, n: int = None):
        if name not in self.graphs: return
        graph = self.graphs[name]

        # FIX: Гарантируем, что не записываем None
        def safe_list(val):
            return val if val is not None else []

        if n is not None:
            if n < len(graph['series']):
                graph['series'][n]['x_data'] = safe_list(x_data_list)
                graph['series'][n]['y_data'] = safe_list(y_data_list)
        else:
            for i, series_item in enumerate(graph['series']):
                if i < len(x_data_list): series_item['x_data'] = safe_list(x_data_list[i])
                if i < len(y_data_list): series_item['y_data'] = safe_list(y_data_list[i])

        if graph['visible']:
            self._redraw_graph_by_timer(name)
            if self.current_hover == name and self.last_x is not None:
                QtCore.QTimer.singleShot(60, lambda: self._update_highlight_and_tooltip(name, self.last_x,
                                                                                        show_tooltip=self.tooltip.isVisible()))

    def _handle_radio_toggle(self, radio, checked):
        if not checked: return
        is_none_radio = (radio == self.none_radio)
        for name in self.checkbox_graphs:
            if name in self.graphs: self.graphs[name]['control'].setEnabled(is_none_radio)
        self._update_visibility()

    def _update_visibility(self):
        active_radio = self.radio_group.checkedButton()
        radio_mode = active_radio is not None and active_radio != self.none_radio
        for name, graph in self.graphs.items():
            should_be_visible = (graph['control'] == active_radio) if radio_mode else (
                    not graph['ct'] and graph['control'].isChecked())
            if should_be_visible != graph['visible']:
                graph['visible'] = should_be_visible
                graph['canvas'].setVisible(should_be_visible)
                if should_be_visible: self._redraw_graph_by_timer(name)
        self.update_graphs_size()

    def _redraw_graph_by_timer(self, name):
        if not self.graphs_timers_flags.get(name):
            self.graphs_timers_flags[name] = True
            QtCore.QTimer.singleShot(50, lambda n=name: self._redraw_event_timer(n))

    def _redraw_event_timer(self, name):
        self._redraw_graph(name)
        self.graphs_timers_flags[name] = False

    def _redraw_graph(self, name):
        if name not in self.graphs: return
        graph = self.graphs[name]
        ax, canvas = graph['ax'], graph['canvas']
        ax.cla()
        ax.set_xlabel(f"{graph['x_name']}, {graph['x_val']}")
        ax.set_ylabel(f"{graph['y_name']}, {graph['y_val']}")
        ax.set_title(name)
        ax.grid(True)
        self.selection_marker = None

        all_x, all_y = [], []

        for series in graph['series']:
            # FIX: Безопасное получение данных. Если None или пусто, пропускаем.
            raw_x = series.get('x_data')
            raw_y = series.get('y_data')

            if raw_x is None or raw_y is None:
                continue

            # Конвертируем в float numpy array, чтобы избежать object array
            x_data = np.array(raw_x, dtype=float)
            y_data = np.array(raw_y, dtype=float)

            series['artists'] = {}

            if x_data.size > 0 and y_data.size > 0:
                all_x.extend(x_data)
                all_y.extend(y_data)
                if series['graph_type'] == 'line':
                    line, = ax.plot(x_data, y_data, '-', color=series['color'], label=series['legend'], zorder=4)
                    points = ax.scatter(x_data, y_data, s=16, c=series['color'], zorder=5)
                    highlight = ax.scatter([], [], s=48, c=series['color'], edgecolors=series['border_color'], lw=2,
                                           zorder=6, visible=False)
                    series['artists'].update({'line': line, 'points': points, 'highlight': highlight})
                elif series['graph_type'] == 'dashed-line':
                    line, = ax.plot(x_data, y_data, '--', color=series['color'], label=series['legend'], zorder=3)
                    series['artists']['line'] = line

        ax.legend()

        # Установка пределов осей
        if all_x and all_y:
            # FIX: используем список, очищенный от None, так что np.min безопасен
            x_min, x_max = np.min(all_x), np.max(all_x)
            y_min, y_max = np.min(all_y), np.max(all_y)

            x_range = x_max - x_min
            y_range = y_max - y_min

            if x_range > 0: ax.set_xlim(x_min - x_range * 0.05, x_max + x_range * 0.05)
            if y_range > 0: ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)

        ax.relim()
        ax.autoscale_view()
        canvas.draw_idle()

    def clear_all_graphs(self):
        for name in list(self.graphs.keys()): self.remove_graph(name)

    def remove_graph(self, name):
        if name not in self.graphs: return
        graph = self.graphs.pop(name)
        if name in self.checkbox_graphs:
            self.checkbox_graphs.remove(name)
            self.checkbox_layout.removeWidget(graph['control'])
        elif name in self.radio_graphs:
            self.radio_graphs.remove(name)
            self.radio_group.removeButton(graph['control'])
            self.radio_layout.removeWidget(graph['control'])
        self.graphs_layout.removeWidget(graph['canvas'])
        graph['canvas'].deleteLater()
        graph['fig'].clf()
        self.update_graphs_size()

    def on_mouse_move(self, event, graph_name):
        graph = self.graphs.get(graph_name)
        if not event.inaxes or event.xdata is None or not graph:
            if self.current_hover: self.on_figure_leave(event, self.current_hover)
            return
        self.current_hover = graph_name
        closest_x, min_dist_x = None, float('inf')

        for series in filter(lambda s: s['graph_type'] == 'line', graph['series']):
            # FIX: Безопасная обработка данных для mouse move
            raw_x = series.get('x_data')
            if raw_x is None: continue

            x_data = np.array(raw_x, dtype=float)
            if x_data.size == 0: continue

            idx = np.argmin(np.abs(x_data - event.xdata))
            dist_x = np.abs(x_data[idx] - event.xdata)
            if dist_x < min_dist_x: min_dist_x, closest_x = dist_x, x_data[idx]

        if closest_x is not None and not np.isclose(closest_x, self.last_x if self.last_x is not None else -np.inf):
            self.last_x = closest_x
            if not self.is_selection_mode:
                self._update_highlight_and_tooltip(graph_name, closest_x)
                self.tooltip_timer.start(500)

    def on_figure_leave(self, event, graph_name):
        graph = self.graphs.get(graph_name)
        needs_redraw = False
        if graph:
            for series in filter(lambda s: s['graph_type'] == 'line', graph['series']):
                highlight_point = series['artists'].get('highlight')
                if highlight_point and highlight_point.get_visible():
                    highlight_point.set_visible(False)
                    needs_redraw = True
        if needs_redraw and graph: graph['canvas'].draw_idle()
        self.current_hover, self.last_x = None, None
        self.tooltip.hide()
        self.tooltip_timer.stop()

    def _update_highlight_and_tooltip(self, graph_name, x_coord, show_tooltip=False):
        graph = self.graphs.get(graph_name)
        if not graph or x_coord is None:
            self.tooltip.hide()
            return
        data_list, needs_redraw = [], False

        for series in filter(lambda s: s['graph_type'] == 'line', graph['series']):
            highlight = series['artists'].get('highlight')
            if not highlight: continue

            # FIX: Безопасная обработка
            raw_x = series.get('x_data')
            raw_y = series.get('y_data')
            if raw_x is None or raw_y is None: continue

            x_data = np.array(raw_x, dtype=float)
            y_data = np.array(raw_y, dtype=float)

            point_found = False
            if x_data.size > 0:
                idx = np.argmin(np.abs(x_data - x_coord))
                if np.isclose(x_data[idx], x_coord):
                    point = (x_data[idx], y_data[idx])
                    highlight.set_offsets(point)
                    if not highlight.get_visible():
                        highlight.set_visible(True)
                        needs_redraw = True
                    data_list.append((series['legend'], y_data[idx], graph['y_val'], series['color']))
                    point_found = True
            if not point_found and highlight.get_visible():
                highlight.set_visible(False)
                needs_redraw = True

        if needs_redraw: graph['canvas'].draw_idle()
        if show_tooltip and data_list:
            pos = QtWidgets.QApplication.desktop().cursor().pos()
            screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
            tip_size = self.tooltip.sizeHint()
            x, y = pos.x() + 16, pos.y() + 16
            if x + tip_size.width() > screen.right(): x = screen.right() - tip_size.width() - 10
            if y + tip_size.height() > screen.bottom(): y = screen.bottom() - tip_size.height() - 10
            self.tooltip.set_group_data(graph['x_name'], x_coord, graph['x_val'], data_list)
            self.tooltip.move(x, y)
            self.tooltip.show()
        else:
            self.tooltip.hide()

    def on_mouse_click(self, event, graph_name):
        if event.button != 1 or not event.inaxes or self.last_x is None: return
        if not self.is_selection_mode:
            # print(f"Клик (не в режиме выбора) в точке X: {self.last_x:.4f}")
            return
        graph = self.graphs.get(graph_name)
        for series in filter(lambda s: s['graph_type'] == 'line', graph['series']):
            # FIX: Безопасная обработка клика
            raw_x = series.get('x_data')
            raw_y = series.get('y_data')
            if raw_x is None or raw_y is None: continue
            x_data = np.array(raw_x, dtype=float)
            y_data = np.array(raw_y, dtype=float)

            if x_data.size > 0:
                idx = np.argmin(np.abs(x_data - self.last_x))
                if np.isclose(x_data[idx], self.last_x):
                    self.point_clicked_signal.emit(graph_name, self.last_x, y_data[idx])
                    return

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.resize_timer_id is not None: self.killTimer(self.resize_timer_id)
        self.resize_timer_id = self.startTimer(100)

    def timerEvent(self, event):
        if event.timerId() == self.resize_timer_id:
            self.killTimer(self.resize_timer_id)
            self.resize_timer_id = None
            self.update_graphs_size()

    def update_graphs_size(self):
        visible_graphs = [g for g in self.graphs.values() if g['visible']]
        if not visible_graphs: return
        available_height = self.scroll_area.viewport().height() - self.control_panel.height() - 8
        if available_height > 0:
            target_height = max(self.MIN_GRAPH_HEIGHT,
                                min(self.MAX_GRAPH_HEIGHT, available_height // len(visible_graphs)))
            for graph in visible_graphs:
                graph['canvas'].setMinimumHeight(target_height)
                graph['canvas'].setMaximumHeight(target_height)
        self.graphs_container.adjustSize()

    def save_all_graphs(self, folder_path):
        """Сохраняет все графики как PNG изображения в указанную папку."""
        import os
        for name, graph in self.graphs.items():
            # Очищаем имя от недопустимых символов
            safe_name = "".join([c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name])
            filename = f"{safe_name}.png"
            full_path = os.path.join(folder_path, filename)

            try:
                # !!! ИСПРАВЛЕНИЕ ЗДЕСЬ !!!
                # Принудительно перерисовываем график перед сохранением,
                # так как скрытые графики могли не получить команды plot()
                self._redraw_graph(name)

                # Принудительно обновляем канвас (рендеринг)
                graph['canvas'].draw()

                # Сохраняем (bbox_inches='tight' убирает лишние белые поля)
                graph['fig'].savefig(full_path, dpi=100, bbox_inches='tight')
            except Exception as e:
                print(f"Error saving graph {name}: {e}")


import colorsys


def generate_plot_colors(names, seed=0, min_contrast=3.0):
    """
    Генерирует цвета для списка графиков (белый фон).
    Вход:
      names - список имён (строки). Порядок сохраняется; дубликаты удаляются, но первое вхождение сохраняется.
      seed - смещение для оттенков (целое или дробное), чтобы получить другой набор цветов при необходимости.
      min_contrast - минимальное отношение контрастности по отношению к белому (по умолчанию 3.0).
    Выход:
      dict {name: "#rrggbb"}
    """
    def hsl_to_hex(h, s, l):
        # colorsys указывает порядок H, L, S
        r, g, b = colorsys.hls_to_rgb(h % 1.0, l, s)
        return '#{0:02x}{1:02x}{2:02x}'.format(int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))

    def relative_luminance(r, g, b):
        # r,g,b в [0,1] в sRGB -> линейная компонента
        def lin(c):
            if c <= 0.03928:
                return c / 12.92
            return ((c + 0.055) / 1.055) ** 2.4
        r_l, g_l, b_l = lin(r), lin(g), lin(b)
        return 0.2126 * r_l + 0.7152 * g_l + 0.0722 * b_l

    def contrast_with_white(hexcolor):
        # Контраст = (L_white + 0.05) / (L_color + 0.05), L_white = 1
        r = int(hexcolor[1:3], 16) / 255.0
        g = int(hexcolor[3:5], 16) / 255.0
        b = int(hexcolor[5:7], 16) / 255.0
        L = relative_luminance(r, g, b)
        return (1.0 + 0.05) / (L + 0.05)

    # Уникальные имена в порядке первого появления
    seen = set()
    unique = []
    for nm in names:
        if nm not in seen:
            seen.add(nm)
            unique.append(nm)

    n = len(unique)
    if n == 0:
        return {}

    colors = {}
    golden_ratio_conj = 0.618033988749895
    # шаблоны насыщенности и светлоты, чтобы цвета отличались не только оттенком
    sats = [0.72, 0.58, 0.85]
    lums = [0.45, 0.55]  # стараться держать светлость относительно низкой, чтобы хорошо читалось на белом фоне

    for i, name in enumerate(unique):
        # равномерное распределение оттенков с небольшим смещением (seed)
        h = (seed + i * golden_ratio_conj) % 1.0
        s = sats[i % len(sats)]
        l = lums[(i // len(sats)) % len(lums)]

        hexc = hsl_to_hex(h, s, l)

        # При необходимости понижаем светлость, чтобы увеличить контраст с белым
        # (умеренно; не делаем полностью чёрным)
        tries = 0
        while contrast_with_white(hexc) < min_contrast and tries < 8:
            l = max(0.15, l - 0.06)  # уменьшаем светлость шагами
            hexc = hsl_to_hex(h, s, l)
            tries += 1

        colors[name] = hexc

    return colors