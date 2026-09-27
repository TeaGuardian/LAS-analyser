# Forest Taxation Benchmark Suite

Personal project: benchmark and comparison of algorithms for individual tree detection from UAV LiDAR (LAS/LAZ) point clouds.

Includes synthetic data generator, several tree detection algorithms, comparison metrics and a PyQt5 GUI for automated benchmarking.

## Features

- Synthetic LAS generator with controllable noise, dropout, crown overlap, type mixing and gaps
- Multiple detection algorithms:
  - CROWN-AXIS
  - CROWN-AXIS Strict
  - CHM Watershed
  - Layer Stack
  - Smart Fusion (V2)
  - Adaptive Fusion (V17)
- Tree matching via Hungarian algorithm
- Detailed metrics: type, height, radius, position + total score
- Full automated benchmark suite with live graphs and Excel export
- Simple LAS cropper utility

## Tech stack

- Python
- NumPy, SciPy, scikit-image, scikit-learn, OpenCV
- laspy
- PyQt5 + Matplotlib
- Pillow

## Quick start

```bash
pip install -r requirements.txt
python app.py