# -*- coding: utf-8 -*-
"""
Simple PyQt5 GUI for cropping a LAS point cloud.

Features
--------
1. **Open LAS** – button opens a *.las* file (using `laspy`).
2. **Top‑view canvas** – Matplotlib canvas embedded in the window shows the
   XY‑projection of the cloud.
3. **Interactive polygon** – left‑click adds a vertex, right‑click removes the
   nearest vertex. The polygon is drawn live.
4. **Crop & Save** – button saves a new LAS file that contains only the points
   lying inside the drawn polygon.

Requirements
------------
pip install pyqt5 matplotlib laspy shapely numpy
"""

import sys
import numpy as np
import laspy
from shapely.geometry import Point, Polygon
from matplotlib.path import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QMessageBox, QHBoxLayout
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class LasCropper(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LAS Cropper")
        self.resize(900, 700)

        # ---------- UI ----------
        open_btn = QPushButton("Open LAS")
        save_btn = QPushButton("Crop & Save")
        open_btn.clicked.connect(self.open_las)
        save_btn.clicked.connect(self.crop_and_save)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(open_btn)
        btn_layout.addWidget(save_btn)

        # Matplotlib canvas
        self.fig = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # Main layout
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(btn_layout)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central)

        # ---------- Data ----------
        self.las = None               # laspy.LasData object
        self.points_xy = None         # Nx2 array of XY coordinates
        self.scatter = None           # Matplotlib scatter handle
        self.poly_vertices = []       # List of (x, y) tuples
        self.poly_line = None         # Matplotlib line handle for polygon

    # --------------------------------------------------------------------- #
    # 1. Load LAS file
    # --------------------------------------------------------------------- #
    def open_las(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open LAS file", "", "LAS files (*.las *.laz)")
        if not path:
            return

        try:
            self.las = laspy.read(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read LAS file:\n{e}")
            return

        # Use X and Y coordinates (top view)
        self.points_xy = np.vstack((self.las.x, self.las.y)).T

        self.ax.clear()
        self.ax.set_title("Top view – left click to add vertex, right click to delete")
        self.scatter = self.ax.scatter(self.points_xy[:, 0], self.points_xy[:, 1],
                                       s=0.2, c='gray', alpha=0.1)
        self.poly_vertices = []
        self.poly_line = None
        self.canvas.draw_idle()

    # --------------------------------------------------------------------- #
    # 2. Mouse interaction – build polygon
    # --------------------------------------------------------------------- #
    def on_click(self, event):
        if self.points_xy is None:
            return  # no data loaded yet

        if event.inaxes != self.ax:
            return

        if event.button == 1:          # left click → add vertex
            self.poly_vertices.append((event.xdata, event.ydata))
        elif event.button == 3:        # right click → remove nearest vertex
            if not self.poly_vertices:
                return
            # Find nearest vertex
            verts = np.array(self.poly_vertices)
            dists = np.hypot(verts[:, 0] - event.xdata, verts[:, 1] - event.ydata)
            idx = np.argmin(dists)
            if dists[idx] < 0.05 * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]):  # tolerance
                self.poly_vertices.pop(idx)

        self._draw_polygon()

    def _draw_polygon(self):
        # Remove previous polygon line
        if self.poly_line:
            self.poly_line.remove()
            self.poly_line = None

        if self.poly_vertices:
            xs, ys = zip(*self.poly_vertices)
            # close the polygon visually
            xs = list(xs) + [xs[0]]
            ys = list(ys) + [ys[0]]
            self.poly_line, = self.ax.plot(xs, ys, 'r-', linewidth=2, marker='o',
                                           markersize=5, markerfacecolor='yellow')
        self.canvas.draw_idle()

    # --------------------------------------------------------------------- #
    # 3. Crop points inside polygon and save
    # --------------------------------------------------------------------- #
    def crop_and_save(self):
        if self.las is None or self.points_xy is None:
            QMessageBox.warning(self, "Warning", "No LAS file loaded.")
            return
        if len(self.poly_vertices) < 3:
            QMessageBox.warning(self, "Warning", "Define a polygon with at least 3 vertices.")
            return

        # Build shapely polygon
        polygon = Polygon(self.poly_vertices)

        # Fast point‑in‑polygon test using matplotlib Path (vectorised)
        path = Path(self.poly_vertices)
        inside_mask = path.contains_points(self.points_xy)

        # Alternative (slower) shapely test:
        # inside_mask = np.array([polygon.contains(Point(p)) for p in self.points_xy])

        if not np.any(inside_mask):
            QMessageBox.information(self, "Info", "No points fall inside the polygon.")
            return

        # Create a new LasData object with only the selected points
        filtered = self.las.points[inside_mask]
        new_las = laspy.LasData(self.las.header)
        new_las.points = filtered

        # Ask for output file name
        out_path, _ = QFileDialog.getSaveFileName(self, "Save cropped LAS", "", "LAS files (*.las *.laz)")
        if not out_path:
            return

        try:
            new_las.write(out_path)
            QMessageBox.information(self, "Success", f"Cropped LAS saved to:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to write LAS file:\n{e}")

    # --------------------------------------------------------------------- #
    # End of class
    # --------------------------------------------------------------------- #


def main():
    app = QApplication(sys.argv)
    win = LasCropper()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
