"""
CLI Script for Evaluating Trained YOLOv8 Fire Detection Model on Validation or Test Sets.
"""

import argparse
from ultralytics import YOLO
from src.config import DATA_YAML_PATH, get_model_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 Fire Detection Model")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DATA_YAML_PATH),
        help="Path to dataset YAML configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=get_model_path(),
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test", "train"],
        help="Dataset split to evaluate on ('val' or 'test')",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size for validation",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold for validation metrics",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading Model: {args.model}")
    model = YOLO(args.model)

    print(
        f"Evaluating model on '{args.split}' split using data config: {args.data}..."
    )
    metrics = model.val(
        data=args.data,
        split=args.split,
        batch=args.batch,
        conf=args.conf,
        plots=True,
    )

    print("=" * 50)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 50)
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"Precision    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")
    print("=" * 50)

    return metrics


if __name__ == "__main__":
    main()
