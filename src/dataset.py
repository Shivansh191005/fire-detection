"""
Dataset Management and Validation Utilities for YOLOv8 Fire Detection.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Dict
from src.config import (
    TRAIN_IMAGES_DIR,
    TRAIN_LABELS_DIR,
    VAL_IMAGES_DIR,
    VAL_LABELS_DIR,
    TEST_IMAGES_DIR,
    TEST_LABELS_DIR,
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def count_files(directory: Path, extensions: tuple = None) -> int:
    """Count matching files in a directory."""
    if not directory.exists():
        return 0
    if extensions:
        return len([f for f in os.listdir(directory) if f.lower().endswith(extensions)])
    return len([f for f in os.listdir(directory) if os.path.isfile(directory / f)])


def get_dataset_stats() -> Dict[str, Dict[str, int]]:
    """Get count of images and labels across train, valid, and test splits."""
    return {
        "Train": {
            "Images": count_files(TRAIN_IMAGES_DIR, IMAGE_EXTENSIONS),
            "Labels": count_files(TRAIN_LABELS_DIR),
        },
        "Valid": {
            "Images": count_files(VAL_IMAGES_DIR, IMAGE_EXTENSIONS),
            "Labels": count_files(VAL_LABELS_DIR),
        },
        "Test": {
            "Images": count_files(TEST_IMAGES_DIR, IMAGE_EXTENSIONS),
            "Labels": count_files(TEST_LABELS_DIR),
        },
    }


def print_dataset_stats() -> None:
    """Print formatted summary table of dataset statistics."""
    stats = get_dataset_stats()
    print("=" * 45)
    print(f"{'Split':<10} | {'Images':<12} | {'Labels':<12}")
    print("-" * 45)
    total_images = 0
    total_labels = 0
    for split, counts in stats.items():
        images = counts["Images"]
        labels = counts["Labels"]
        total_images += images
        total_labels += labels
        print(f"{split:<10} | {images:<12} | {labels:<12}")
    print("-" * 45)
    print(f"{'Total':<10} | {total_images:<12} | {total_labels:<12}")
    print("=" * 45)


def split_train_val(val_ratio: float = 0.20, seed: int = 42) -> int:
    """
    Split a subset of training images/labels into validation directory.
    Returns the number of images moved.
    """
    random.seed(seed)
    os.makedirs(VAL_IMAGES_DIR, exist_ok=True)
    os.makedirs(VAL_LABELS_DIR, exist_ok=True)

    images = [
        img
        for img in os.listdir(TRAIN_IMAGES_DIR)
        if img.lower().endswith(IMAGE_EXTENSIONS)
    ]

    num_valid = int(val_ratio * len(images))
    if num_valid == 0:
        print("No images to move.")
        return 0

    random.shuffle(images)
    valid_set = images[:num_valid]

    moved = 0
    for img in valid_set:
        src_img = TRAIN_IMAGES_DIR / img
        dst_img = VAL_IMAGES_DIR / img
        shutil.move(src_img, dst_img)

        label_name = os.path.splitext(img)[0] + ".txt"
        src_label = TRAIN_LABELS_DIR / label_name
        dst_label = VAL_LABELS_DIR / label_name
        if src_label.exists():
            shutil.move(src_label, dst_label)

        moved += 1

    print(f"Successfully moved {moved} images to validation set.")
    return moved


if __name__ == "__main__":
    print_dataset_stats()
