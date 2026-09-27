import sys
import time
import os
import shutil
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QDoubleSpinBox, QSpinBox, QPushButton,
    QProgressBar, QScrollArea, QFrame, QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, pyqtSlot, QTimer

import backend
import utils

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PARSER_COLORS = utils.generate_plot_colors(list(backend.PARSERS.keys()))

# Стили кнопок (оставляем как было)
BTN_STYLE_START = "QPushButton { background-color: #28a745; color: white; font-weight: bold; padding: 8px; border-radius: 4px; border: 1px solid #218838; } QPushButton:hover { background-color: #218838; } QPushButton:disabled { background-color: #94d3a2; color: #e8f5e9; border: 1px solid #94d3a2; }"
BTN_STYLE_STOP = "QPushButton { background-color: #dc3545; color: white; font-weight: bold; padding: 8px; border-radius: 4px; border: 1px solid #c82333; } QPushButton:hover { background-color: #c82333; } QPushButton:disabled { background-color: #eeb0b7; color: #ffebee; border: 1px solid #eeb0b7; }"
BTN_STYLE_EXPORT = "QPushButton { background-color: #007bff; color: white; font-weight: bold; padding: 10px; margin-top: 5px; border-radius: 4px; border: 1px solid #0069d9; } QPushButton:hover { background-color: #0069d9; } QPushButton:disabled { background-color: #80bdff; color: #e3f2fd; border: 1px solid #80bdff; }"


class BenchmarkWorker(QObject):
    sig_global_progress = pyqtSignal(int, int)
    sig_task_status = pyqtSignal(str, str, int, bool)
    sig_result_ready = pyqtSignal(dict)
    sig_finished = pyqtSignal()

    def __init__(self, step, gen_workers, parse_workers, comp_workers, area_size, las_storage, img_storage):
        super().__init__()
        self.step = step
        self.gen_workers = gen_workers
        self.parse_workers = parse_workers
        self.comp_workers = comp_workers
        self.area_size = area_size
        self.las_storage = las_storage
        self.img_storage = img_storage
        self.manager = None
        self._is_running = False

    @pyqtSlot()
    def start_benchmark(self):
        self._is_running = True
        self.manager = utils.BenchmarkManager(
            las_path=self.las_storage, img_path=self.img_storage,
            gen_workers=self.gen_workers, parse_workers=self.parse_workers,
            compare_workers=self.comp_workers, area_size=self.area_size
        )
        self.manager.on_global_progress = self.sig_global_progress.emit
        self.manager.on_task_status = self.sig_task_status.emit
        self.manager.on_result = self.sig_result_ready.emit

        def on_manager_finished(results): self._is_running = False

        self.manager.on_finished = on_manager_finished

        self.manager.start(step=self.step)
        while self.manager._is_running and self._is_running: QThread.msleep(100)
        self.sig_finished.emit()

    def stop(self):
        self._is_running = False
        if self.manager: self.manager.stop()


class TaskWidget(QWidget):
    def __init__(self, name, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self);
        layout.setContentsMargins(5, 5, 5, 5);
        layout.setSpacing(2)
        self.lbl_name = QLabel(name);
        self.lbl_name.setStyleSheet("font-weight: bold; font-size: 11px;")
        self.pbar = QProgressBar();
        self.pbar.setFixedHeight(8);
        self.pbar.setTextVisible(False)
        layout.addWidget(self.lbl_name);
        layout.addWidget(self.pbar)
        self.setStyleSheet("background-color: #F9F9F9; border: 1px solid #DDDDDD; border-radius: 4px;")

    def update_progress(self, val): self.pbar.setValue(val)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Forest Taxation Benchmark Suite")
        self.resize(1280, 850)

        self.las_storage_path = "B:/storage/las" if os.path.exists("B:/") else "storage/las"
        self.img_storage_path = "storage/img"

        # 1. Основные графики (Параметр -> Total Score)
        self.graph_data = {
            "noise_level": defaultdict(lambda: defaultdict(list)),
            "dropout_rate": defaultdict(lambda: defaultdict(list)),
            "overlap_factor": defaultdict(lambda: defaultdict(list)),
            "mixing_factor": defaultdict(lambda: defaultdict(list)),
            "gap_factor": defaultdict(lambda: defaultdict(list)),
        }
        # 2. График времени (Complexity -> Duration)
        self.graph_time_data = defaultdict(lambda: defaultdict(list))

        # 3. Детальные метрики (Metric -> Parser -> Complexity -> [Scores])
        self.stats_data = {
            "type": defaultdict(lambda: defaultdict(list)),
            "height": defaultdict(lambda: defaultdict(list)),
            "radius": defaultdict(lambda: defaultdict(list)),
            "position": defaultdict(lambda: defaultdict(list))
        }

        self.active_tasks = {}
        self.start_time = None;
        self.completed_ops = 0;
        self.total_ops = 1

        self.init_ui()
        self.monitor_timer = QTimer();
        self.monitor_timer.timeout.connect(self.on_monitor_tick)
        self.worker_thread = None;
        self.worker = None;
        self.on_monitor_tick()

    def init_ui(self):
        central_widget = QWidget();
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = QWidget();
        left_layout = QVBoxLayout(left_panel);
        left_panel.setFixedWidth(360)

        grp_settings = QGroupBox("Настройки теста");
        form_layout = QVBoxLayout(grp_settings)
        lbl_size = QLabel("Размер области (X, Y, Z):");
        size_layout = QHBoxLayout()
        self.spin_size_x = QSpinBox();
        self.spin_size_x.setRange(10, 5000);
        self.spin_size_x.setValue(100);
        self.spin_size_x.setSuffix(" м")
        self.spin_size_y = QSpinBox();
        self.spin_size_y.setRange(10, 5000);
        self.spin_size_y.setValue(100);
        self.spin_size_y.setSuffix(" м")
        self.spin_size_z = QSpinBox();
        self.spin_size_z.setRange(10, 500);
        self.spin_size_z.setValue(50);
        self.spin_size_z.setSuffix(" м")
        size_layout.addWidget(self.spin_size_x);
        size_layout.addWidget(self.spin_size_y);
        size_layout.addWidget(self.spin_size_z)

        lbl_step = QLabel("Шаг параметров (0.01 - 0.5):")
        self.spin_step = QDoubleSpinBox();
        self.spin_step.setRange(0.01, 0.5);
        self.spin_step.setSingleStep(0.05);
        self.spin_step.setValue(0.1)

        lbl_workers = QLabel("Распределение потоков:")
        self.spin_gen = self._create_worker_spin("Generators:", 2)
        self.spin_parse = self._create_worker_spin("Parsers:", 4)
        self.spin_comp = self._create_worker_spin("Comparators:", 2)

        form_layout.addWidget(lbl_size);
        form_layout.addLayout(size_layout)
        form_layout.addWidget(lbl_step);
        form_layout.addWidget(self.spin_step)
        form_layout.addWidget(lbl_workers);
        form_layout.addWidget(self.spin_gen)
        form_layout.addWidget(self.spin_parse);
        form_layout.addWidget(self.spin_comp)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("ЗАПУСК");
        self.btn_start.setStyleSheet(BTN_STYLE_START);
        self.btn_start.clicked.connect(self.start_benchmark)
        self.btn_stop = QPushButton("СТОП");
        self.btn_stop.setStyleSheet(BTN_STYLE_STOP);
        self.btn_stop.setEnabled(False);
        self.btn_stop.clicked.connect(self.stop_benchmark)
        btn_layout.addWidget(self.btn_start);
        btn_layout.addWidget(self.btn_stop)
        form_layout.addLayout(btn_layout)

        self.btn_save = QPushButton("ЭКСПОРТ ОТЧЕТА");
        self.btn_save.setStyleSheet(BTN_STYLE_EXPORT);
        self.btn_save.clicked.connect(self.export_results)
        form_layout.addWidget(self.btn_save)
        left_layout.addWidget(grp_settings)

        grp_disk = QGroupBox("Мониторинг диска");
        disk_layout = QVBoxLayout(grp_disk)
        self.lbl_disk_las = QLabel("LAS Storage:");
        self.pbar_disk_las = QProgressBar();
        self.pbar_disk_las.setTextVisible(True)
        self.lbl_disk_img = QLabel("IMG Storage:");
        self.pbar_disk_img = QProgressBar();
        self.pbar_disk_img.setTextVisible(True)
        disk_layout.addWidget(self.lbl_disk_las);
        disk_layout.addWidget(self.pbar_disk_las)
        disk_layout.addWidget(self.lbl_disk_img);
        disk_layout.addWidget(self.pbar_disk_img)
        left_layout.addWidget(grp_disk)

        grp_status = QGroupBox("Прогресс выполнения");
        status_layout = QVBoxLayout(grp_status)
        self.lbl_global_status = QLabel("Готов к работе");
        self.pbar_global = QProgressBar()
        self.lbl_time_info = QLabel("Время: 00:00:00 | Осталось: --:--:--");
        self.lbl_time_info.setAlignment(Qt.AlignCenter);
        self.lbl_time_info.setStyleSheet("color: #666; font-family: monospace;")
        status_layout.addWidget(self.lbl_global_status);
        status_layout.addWidget(self.pbar_global);
        status_layout.addWidget(self.lbl_time_info)
        left_layout.addWidget(grp_status)

        grp_tasks = QGroupBox("Активные потоки");
        tasks_layout = QVBoxLayout(grp_tasks)
        self.scroll_area = QScrollArea();
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget();
        self.tasks_list_layout = QVBoxLayout(self.scroll_content);
        self.tasks_list_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.scroll_content);
        tasks_layout.addWidget(self.scroll_area)
        left_layout.addWidget(grp_tasks)

        self.graph_widget = utils.GraphWidget()
        self._init_graphs()

        main_layout.addWidget(left_panel);
        main_layout.addWidget(self.graph_widget, stretch=1)

    def _create_worker_spin(self, text, default):
        w = QWidget();
        l = QHBoxLayout(w);
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(QLabel(text));
        spin = QSpinBox();
        spin.setRange(0, 16);
        spin.setValue(default)
        l.addWidget(spin);
        return w

    def _init_graphs(self):
        parsers = sorted(backend.PARSERS.keys())
        colors = [PARSER_COLORS.get(p, "#333333") for p in parsers]

        self.params_config = [
            ("noise_level", "Шум", "Уровень шума (0-1)"),
            ("dropout_rate", "Выпадение точек", "Доля выпадения"),
            ("overlap_factor", "Перекрытие крон", "Коэф. перекрытия"),
            ("mixing_factor", "Смешение типов", "Коэф. смешения"),
            ("gap_factor", "Пустоты леса", "Коэф. пустот"),
        ]

        # 1. Графики по отдельным параметрам (Total Score)
        for _, title, x_label in self.params_config:
            self.graph_widget.add_graph(name=title, x_name=x_label, y_name="Total Score", x_val="", y_val="", ct=False,
                                        colors=colors, legends=parsers, graph_types=['line'] * len(parsers),
                                        approximations=[True] * len(parsers))

        # 2. График времени
        self.graph_widget.add_graph(name="Скорость обработки", x_name="Сложность (Сумма параметров)", y_name="Время",
                                    x_val="", y_val="сек", ct=False, colors=colors, legends=parsers,
                                    graph_types=['line'] * len(parsers), approximations=[True] * len(parsers))

        # 3. Новые детальные графики (Metric vs Complexity)
        # Мы используем "Сложность" (Complexity) как ось X
        metrics_map = {
            "type": "Точность: Тип дерева",
            "height": "Точность: Высота",
            "radius": "Точность: Радиус кроны",
            "position": "Точность: Позиция (XY)"
        }

        for metric_key, title in metrics_map.items():
            self.graph_widget.add_graph(
                name=title,
                x_name="Сложность данных",
                y_name=f"{metric_key.capitalize()} Score",
                x_val="idx", y_val="0-100",
                ct=False, colors=colors, legends=parsers,
                graph_types=['line'] * len(parsers), approximations=[True] * len(parsers)
            )

        # Дефолтный график
        default_title = next(t for k, t, _ in self.params_config if k == "noise_level")
        """if default_title in self.graph_widget.graphs:
            self.graph_widget.graphs[default_title]['visible'] = True
            self.graph_widget.graphs[default_title]['control'].setChecked(True)
            self.graph_widget._update_visibility()"""

    def on_monitor_tick(self):
        for path_str, pbar, label in [(self.las_storage_path, self.pbar_disk_las, self.lbl_disk_las),
                                      (self.img_storage_path, self.pbar_disk_img, self.lbl_disk_img)]:
            try:
                check_path = path_str if os.path.exists(path_str) else "."
                if not os.path.exists(path_str) and os.path.splitdrive(path_str)[0]: check_path = \
                os.path.splitdrive(path_str)[0]
                usage = shutil.disk_usage(check_path)
                percent = int((usage.used / usage.total) * 100);
                free_gb = usage.free / (1024 ** 3)
                pbar.setValue(percent);
                pbar.setFormat(f"%p% (Своб: {free_gb:.1f} GB)")
                style = "#dc3545" if percent > 90 else "#ffc107" if percent > 75 else "#28a745"
                pbar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {style}; }}")
                label.setToolTip(f"Path: {os.path.abspath(path_str)}")
            except:
                pbar.setValue(0); pbar.setFormat("Error")
        if self.worker and self.start_time:
            elapsed = time.time() - self.start_time
            remaining_str = "--:--:--"
            if self.completed_ops > 0 and self.total_ops > 0:
                left_ops = self.total_ops - self.completed_ops
                if left_ops > 0: remaining_str = time.strftime('%H:%M:%S',
                                                               time.gmtime(left_ops * (elapsed / self.completed_ops)))
            self.lbl_time_info.setText(
                f"Время: {time.strftime('%H:%M:%S', time.gmtime(elapsed))} | Осталось: {remaining_str}")

    def start_benchmark(self):
        self.btn_start.setEnabled(False);
        self.btn_stop.setEnabled(True);
        self.btn_save.setEnabled(False)
        self.active_tasks.clear()
        while self.tasks_list_layout.count():
            if self.tasks_list_layout.takeAt(0).widget(): self.tasks_list_layout.takeAt(0).widget().deleteLater()

        # Очистка всех данных
        for p_key in self.graph_data:
            for p in self.graph_data[p_key]: self.graph_data[p_key][p].clear()
        for p in self.graph_time_data: self.graph_time_data[p].clear()
        for m in self.stats_data:
            for p in self.stats_data[m]: self.stats_data[m][p].clear()

        self.graph_widget.clear_all_graphs()
        self._init_graphs()
        self.pbar_global.setValue(0);
        self.lbl_global_status.setText("Инициализация...")
        self.start_time = time.time();
        self.completed_ops = 0;
        self.total_ops = 1;
        self.monitor_timer.start(1000)

        area = (self.spin_size_x.value(), self.spin_size_y.value(), self.spin_size_z.value())
        self.worker_thread = QThread()
        self.worker = BenchmarkWorker(self.spin_step.value(), self.spin_gen.findChild(QSpinBox).value(),
                                      self.spin_parse.findChild(QSpinBox).value(),
                                      self.spin_comp.findChild(QSpinBox).value(), area, self.las_storage_path,
                                      self.img_storage_path)
        self.worker.moveToThread(self.worker_thread);
        self.worker_thread.started.connect(self.worker.start_benchmark)
        self.worker.sig_global_progress.connect(self.update_global_progress);
        self.worker.sig_task_status.connect(self.update_task_status)
        self.worker.sig_result_ready.connect(self.handle_result);
        self.worker.sig_finished.connect(self.on_benchmark_finished)
        self.worker_thread.start()

    def stop_benchmark(self):
        if self.worker: self.worker.stop(); self.lbl_global_status.setText("Остановка..."); self.btn_start.setEnabled(
            False); self.btn_stop.setEnabled(False)

    def on_benchmark_finished(self):
        self.monitor_timer.stop();
        self.worker_thread.quit();
        self.worker_thread.wait();
        self.worker = None;
        self.worker_thread = None
        self.btn_start.setEnabled(True);
        self.btn_stop.setEnabled(False);
        self.btn_save.setEnabled(True)
        self.lbl_global_status.setText("Тестирование завершено");
        self.on_monitor_tick()
        QMessageBox.information(self, "Успех", "Тестирование завершено!")

    def export_results(self):
        directory = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения результатов")
        if not directory: return
        try:
            # Лист 1: Quality (по параметрам)
            q_rows = []
            for param, p_dict in self.graph_data.items():
                for parser, v_dict in p_dict.items():
                    for x, scores in v_dict.items():
                        if scores: q_rows.append({"Param": param, "Parser": parser, "Val": x, "Score": np.mean(scores)})

            # Лист 2: Time
            t_rows = []
            for parser, v_dict in self.graph_time_data.items():
                for x, vals in v_dict.items():
                    if vals: t_rows.append({"Param": "Complexity", "Parser": parser, "Val": x, "Time": np.mean(vals)})

            # Лист 3: Detailed Stats
            s_rows = []
            for metric, p_dict in self.stats_data.items():
                for parser, v_dict in p_dict.items():
                    for x, scores in v_dict.items():
                        if scores: s_rows.append(
                            {"Metric": metric, "Parser": parser, "Complexity": x, "Score": np.mean(scores)})

            with pd.ExcelWriter(os.path.join(directory, "benchmark_results.xlsx")) as writer:
                pd.DataFrame(q_rows).to_excel(writer, sheet_name="General Quality", index=False)
                pd.DataFrame(t_rows).to_excel(writer, sheet_name="Time", index=False)
                pd.DataFrame(s_rows).to_excel(writer, sheet_name="Component Stats", index=False)

            self.graph_widget.save_all_graphs(directory)
            QMessageBox.information(self, "Экспорт", f"Сохранено в:\n{directory}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    @pyqtSlot(int, int)
    def update_global_progress(self, c, t):
        self.completed_ops = c;
        self.total_ops = t
        if t > 0: self.pbar_global.setValue(int((c / t) * 100)); self.lbl_global_status.setText(f"Выполнено {c} / {t}")

    @pyqtSlot(str, str, int, bool)
    def update_task_status(self, uid, name, p, f):
        if f:
            if uid in self.active_tasks: self.active_tasks.pop(uid).deleteLater()
        else:
            if uid not in self.active_tasks: self.active_tasks[uid] = TaskWidget(
                name); self.tasks_list_layout.insertWidget(0, self.active_tasks[uid])
            self.active_tasks[uid].lbl_name.setText(name);
            self.active_tasks[uid].update_progress(p)

    @pyqtSlot(dict)
    def handle_result(self, result):
        parser = result['parser']
        # 'score' - это total_score (float)
        total_score = result['score']
        # 'stats' - это dict {'total': ..., 'type': ..., ...}
        stats = result['stats']
        params = result['params']
        duration = result.get('duration', 0.0)

        # 1. Считаем коэффициент сложности (Сумма всех параметров)
        # Так как генератор меняет только один параметр за раз, это и будет значением оси X
        complexity = sum([
            params['noise_level'], params['dropout_rate'],
            params['overlap_factor'], params['mixing_factor'],
            params['gap_factor']
        ])

        # 2. Обновляем старые графики (Зависимость Total Score от конкретного параметра)
        for key, title, _ in self.params_config:
            val = params[key]
            self.graph_data[key][parser][val].append(total_score)
            self._update_single_graph_data(title, self.graph_data[key])

        # 3. Обновляем график времени (от сложности)
        self.graph_time_data[parser][complexity].append(duration)
        self._update_single_graph_data("Скорость обработки", self.graph_time_data)

        # 4. Обновляем новые детальные графики (Метрика от сложности)
        # metrics_map: {'type': "Точность: Тип дерева", ...}
        metrics_map = {
            "type": "Точность: Тип дерева",
            "height": "Точность: Высота",
            "radius": "Точность: Радиус кроны",
            "position": "Точность: Позиция (XY)"
        }

        for metric_key, graph_title in metrics_map.items():
            if metric_key in stats:
                val = stats[metric_key]
                self.stats_data[metric_key][parser][complexity].append(val)
                self._update_single_graph_data(graph_title, self.stats_data[metric_key])

    def _update_single_graph_data(self, graph_title, data_source):
        parsers = sorted(backend.PARSERS.keys())
        x_list = []
        y_list = []
        for p in parsers:
            p_data = data_source[p]
            if not p_data: x_list.append([]); y_list.append([]); continue
            sorted_x = sorted(p_data.keys())
            x_list.append(sorted_x);
            y_list.append([np.mean(p_data[x]) for x in sorted_x])
        self.graph_widget.set_data(graph_title, x_list, y_list)


def except_hook(cls, exception, traceback): sys.__excepthook__(cls, exception, traceback)


if __name__ == "__main__":
    sys.excepthook = except_hook
    app = QApplication(sys.argv)
    font = app.font()
    font.setFamily("Segoe UI")
    font.setPointSize(9)
    app.setFont(font)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())