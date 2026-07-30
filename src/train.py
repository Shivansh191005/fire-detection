"""
CLI Script for Training YOLOv8 on the Fire Detection Dataset.
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
from src.config import DATA_YAML_PATH, get_default_config, get_model_path
from src.utils import print_device_info


def parse_args():
    cfg = get_default_config().get("train", {})
    parser = argparse.ArgumentParser(description="Train YOLOv8 Fire Detection Model")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DATA_YAML_PATH),
        help="Path to dataset YAML file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=get_model_path(),
        help="Pretrained weights or checkpoint path",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=cfg.get("epochs", 40),
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=cfg.get("batch", 8),
        help="Batch size for training",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=cfg.get("imgsz", 640),
        help="Input image size (pixels)",
    )
    parser.add_argument(
        "--device",
        default=cfg.get("device", 0),
        help="CUDA device index (0) or 'cpu'",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cfg.get("workers", 0),
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=cfg.get("name", "YOLOv8s_Fire"),
        help="Run name for saved model",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=cfg.get("project", "Fire_Detection"),
        help="Project directory name for saving runs",
    )
    return parser.parse_args()


def main():
    print_device_info()
    args = parse_args()

    print(f"Loading YOLOv8 Model: {args.model}")
    model = YOLO(args.model)

    print(
        f"Starting training on dataset {args.data} for {args.epochs} epochs..."
    )
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    print("Training Complete!")
    return results


if __name__ == "__main__":
    main()
