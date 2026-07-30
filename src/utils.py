"""
Utility Functions for Fire Detection YOLOv8 Project.
"""

import os
from pathlib import Path
import torch


def get_device_info() -> dict:
    """Check PyTorch CUDA availability and return device information."""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": "CPU",
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
    return info


def print_device_info() -> None:
    """Print PyTorch and hardware device details."""
    info = get_device_info()
    print("=" * 45)
    print(f"PyTorch Version  : {info['pytorch_version']}")
    print(f"CUDA Available   : {info['cuda_available']}")
    print(f"Active Device    : {info['device_name']}")
    print("=" * 45)


def check_file_exists(file_path: str) -> bool:
    """Check whether a file path exists."""
    return os.path.exists(file_path)
