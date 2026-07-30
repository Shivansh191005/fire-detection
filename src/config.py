"""
Centralized Configuration for Fire Detection YOLOv8 Project.
"""

import os
from pathlib import Path
import yaml

# Directories
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT_DIR / "configs"
RUNS_DIR = ROOT_DIR / "runs"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# Dataset Directories
TRAIN_IMAGES_DIR = ROOT_DIR / "train" / "images"
TRAIN_LABELS_DIR = ROOT_DIR / "train" / "labels"
VAL_IMAGES_DIR = ROOT_DIR / "valid" / "images"
VAL_LABELS_DIR = ROOT_DIR / "valid" / "labels"
TEST_IMAGES_DIR = ROOT_DIR / "test" / "images"
TEST_LABELS_DIR = ROOT_DIR / "test" / "labels"

# YAML Config Files
DATA_YAML_PATH = CONFIGS_DIR / "data.yaml"
DEFAULT_YAML_PATH = CONFIGS_DIR / "default.yaml"

# Default Model Checkpoints
PRETRAINED_MODEL = ROOT_DIR / "yolov8s.pt"
TRAINED_BEST_MODEL = RUNS_DIR / "detect" / "Fire_Detection" / "YOLOv8s_Fire" / "weights" / "best.pt"


def get_default_config() -> dict:
    """Load default hyperparameters from configs/default.yaml."""
    if not DEFAULT_YAML_PATH.exists():
        return {}
    with open(DEFAULT_YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_path(custom_path: str = None) -> str:
    """
    Returns the path to the model weights to use.
    Prioritizes custom_path if provided, then trained best.pt, then yolov8s.pt.
    """
    if custom_path and os.path.exists(custom_path):
        return custom_path
    if TRAINED_BEST_MODEL.exists():
        return str(TRAINED_BEST_MODEL)
    if PRETRAINED_MODEL.exists():
        return str(PRETRAINED_MODEL)
    return "yolov8s.pt"
