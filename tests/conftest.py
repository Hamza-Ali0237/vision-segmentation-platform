"""
Shared pytest fixtures used across all test modules.

All fixtures are CPU-only and use tiny tensors / synthetic data so they
run in seconds with no GPU and no real dataset required.
"""

import os
import sys
import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Make sure src/ is importable when running pytest from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Minimal config dict that mirrors base.yaml structure."""
    return {
        "project": {"name": "test", "seed": 42},
        "paths": {
            "data_dir":   "/tmp/test_data",
            "model_dir":  "/tmp/test_models",
            "output_dir": "/tmp/test_output",
        },
        "data": {
            "image_size":  64,   # tiny — keeps tests fast
            "in_channels": 3,
            "num_workers": 0,    # 0 = main process, no multiprocessing in tests
        },
        "model": {"num_classes": 1, "architectures": ["unet", "segnet"]},
        "training": {
            "epochs":                  2,
            "batch_size":              2,
            "learning_rate":           1e-4,
            "weight_decay":            1e-4,
            "weight_bce":              1.0,
            "weight_dice":             1.0,
            "lr_patience":             5,
            "lr_factor":               0.1,
            "early_stopping_patience": 3,
        },
        "augmentation": {"apply": False, "rotate_limit": 10},
        "mlflow": {"experiment_name": "test-experiment", "artifact_location": ""},
        "optuna": {
            "n_trials": 2,
            "direction": "minimize",
            "pruning": False,
            "search_space": {
                "lr":           [1e-4, 1e-3],
                "weight_decay": [1e-5, 1e-4],
                "weight_bce":   [0.8, 1.2],
                "weight_dice":  [0.8, 1.2],
                "batch_size":   [2],
                "lr_patience":  [5],
            },
        },
    }


# ---------------------------------------------------------------------------
# Device fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Tensor fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_images():
    """(B=2, C=3, H=64, W=64) float32 tensor — simulates a loader batch."""
    return torch.randn(2, 3, 64, 64)


@pytest.fixture
def batch_masks():
    """(B=2, 1, H=64, W=64) binary float32 tensor."""
    return (torch.rand(2, 1, 64, 64) > 0.5).float()


# ---------------------------------------------------------------------------
# Synthetic dataset on disk
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_dataset_dirs(tmp_path):
    """
    Creates a tiny synthetic image/mask dataset on disk.
    Returns (images_dir, masks_dir).
    """
    images_dir = tmp_path / "images"
    masks_dir  = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    for i in range(6):   # 6 samples → 4 train / 1 val / 1 test
        fname = f"sample_{i:04d}.png"

        # Random RGB image
        img_arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(img_arr).save(images_dir / fname)

        # Binary mask
        mask_arr = (np.random.rand(64, 64) > 0.5).astype(np.uint8) * 255
        Image.fromarray(mask_arr, mode="L").save(masks_dir / fname)

    return str(images_dir), str(masks_dir)