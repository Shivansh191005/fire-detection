# 🔥 Real-Time Fire Detection with YOLOv8

[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-blue.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0.0-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, modular Python framework and demonstration suite for **Real-Time Fire Detection** using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics). Designed for high-speed inference on edge devices, GPUs, and video streams.

---

## 🚀 Overview

Early fire detection is critical for preventing disasters in industrial plants, forests, and residential buildings. This repository provides a complete, well-structured deep learning pipeline powered by **YOLOv8s**, fine-tuned specifically to detect flames and fire outbreaks in diverse environments.

### Key Features
- **High Accuracy & Real-Time Speed**: Combines YOLOv8's state-of-the-art detector with CUDA GPU acceleration for rapid inference (~4ms per frame).
- **Modular Codebase**: Organized Python modules (`src/`) for clean separation of dataset validation, training, prediction, and evaluation.
- **CLI Tools**: Full command-line interface (`python -m src.train`, `python -m src.predict`, `python -m src.evaluate`).
- **Interactive Demos**: Ready-to-run Jupyter notebook (`notebooks/fire_detection_demo.ipynb`).
- **GitHub & Production Ready**: Configured with strict `.gitignore` to prevent repository bloating from large videos or datasets.

---

## 📊 Model Performance

After fine-tuning **YOLOv8s** for **40 epochs** on the Fire Detection dataset, the model achieved the following validation metrics:

| Metric | Value |
| :--- | :---: |
| **mAP@0.5** | `0.6740` |
| **mAP@0.5:0.95** | `0.3660` |
| **Precision** | `0.6910` |
| **Recall** | `0.6340` |
| **Inference Speed (RTX 4050)** | `4.1 ms/frame` |

---

## 📂 Repository Structure

```text
fire-detection-yolov8/
├── configs/
│   ├── data.yaml                 # YOLOv8 dataset configuration file
│   └── default.yaml              # Default training and inference hyperparameters
├── notebooks/
│   └── fire_detection_demo.ipynb # Interactive demo and exploration notebook
├── src/
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Centralized project paths and settings
│   ├── dataset.py                # Dataset stats, validation, and splitting utilities
│   ├── train.py                  # Modular training CLI script
│   ├── predict.py                # Video / Image / Webcam inference CLI script
│   ├── evaluate.py               # Validation and mAP metric evaluation CLI script
│   └── utils.py                  # PyTorch hardware checks and helpers
├── .gitignore                    # Prevents datasets and large binaries from bloating Git
├── pyproject.toml                # Modern Python project packaging metadata
├── requirements.txt              # Dependency specification
├── LICENSE                       # MIT License
└── README.md                     # Project documentation
```

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/arorashivansh/fire-detection-yolov8.git
   cd fire-detection-yolov8
   ```

2. **Create a Virtual Environment (Optional but Recommended)**:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚡ Quickstart Usage

### 1. Check Hardware & Dataset Statistics
Run the dataset utility to inspect image and label distribution:
```bash
python -m src.dataset
```

### 2. Training the Model
Fine-tune a YOLOv8 model on your custom fire dataset:
```bash
python -m src.train --epochs 40 --batch 8 --imgsz 640 --device 0
```

### 3. Running Real-Time Prediction
Run inference on a video, directory of images, or webcam (use `--source 0` for webcam):
```bash
python -m src.predict --source path/to/video.mp4 --conf 0.35 --save
```

### 4. Evaluating Model Metrics
Validate the trained weights against the validation set:
```bash
python -m src.evaluate --split val
```

---

## 📓 Jupyter Notebook Demo

Explore the complete pipeline interactively using Jupyter Notebook or VS Code:
```bash
jupyter notebook notebooks/fire_detection_demo.ipynb
```

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
