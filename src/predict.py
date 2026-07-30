"""
CLI Script for Running Inference with YOLOv8 on Images, Videos, or Webcams.
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO
from src.config import get_model_path, get_default_config, ROOT_DIR


def get_default_source() -> str:
    """Find a default test video or image directory."""
    default_video = ROOT_DIR / "5622766-uhd_3840_2160_30fps.mp4"
    if default_video.exists():
        return str(default_video)
    test_img_dir = ROOT_DIR / "test" / "images"
    if test_img_dir.exists():
        return str(test_img_dir)
    return "0"  # Fallback to webcam index 0


def parse_args():
    cfg = get_default_config().get("predict", {})
    parser = argparse.ArgumentParser(description="Run YOLOv8 Fire Detection Inference")
    parser.add_argument(
        "--source",
        type=str,
        default=get_default_source(),
        help="Path to input video, image directory, or webcam index (0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=get_model_path(),
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=cfg.get("conf", 0.35),
        help="Confidence threshold for detection",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=cfg.get("iou", 0.70),
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display live detections in an OpenCV GUI window",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save output annotated files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading Model: {args.model}")
    model = YOLO(args.model)

    save_flag = not args.no_save
    print(
        f"Running inference on source '{args.source}' (conf={args.conf}, iou={args.iou}, save={save_flag})..."
    )

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        show=args.show,
        save=save_flag,
        stream=False,
    )

    print("Inference completed successfully.")
    return results


if __name__ == "__main__":
    main()
