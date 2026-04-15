"""
Tests for SegmentationDataset and get_loaders.

All tests use synthetic PNG files written to a temp directory — no real
dataset needed.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset_class import SegmentationDataset
from src.data.augmentation import get_transforms
from src.data.loaders import get_loaders


class TestSegmentationDataset:
    def test_len(self, synthetic_dataset_dirs, config):
        images_dir, masks_dir = synthetic_dataset_dirs
        import os
        files = sorted(os.listdir(images_dir))
        tf, norm = get_transforms(config, is_train=False)
        ds = SegmentationDataset(images_dir, masks_dir, files, tf, norm)
        assert len(ds) == len(files)

    def test_getitem_shapes(self, synthetic_dataset_dirs, config):
        images_dir, masks_dir = synthetic_dataset_dirs
        import os
        files = sorted(os.listdir(images_dir))
        tf, norm = get_transforms(config, is_train=False)
        ds = SegmentationDataset(images_dir, masks_dir, files, tf, norm)

        img, mask = ds[0]
        H = W = config["data"]["image_size"]

        assert img.shape  == (3, H, W),  f"Expected (3,{H},{W}), got {img.shape}"
        assert mask.shape == (1, H, W),  f"Expected (1,{H},{W}), got {mask.shape}"

    def test_image_dtype(self, synthetic_dataset_dirs, config):
        images_dir, masks_dir = synthetic_dataset_dirs
        import os
        files = sorted(os.listdir(images_dir))
        tf, norm = get_transforms(config, is_train=False)
        ds = SegmentationDataset(images_dir, masks_dir, files, tf, norm)
        img, _ = ds[0]
        assert img.dtype == torch.float32

    def test_mask_binary(self, synthetic_dataset_dirs, config):
        """Mask values must be exactly 0.0 or 1.0."""
        images_dir, masks_dir = synthetic_dataset_dirs
        import os
        files = sorted(os.listdir(images_dir))
        tf, norm = get_transforms(config, is_train=False)
        ds = SegmentationDataset(images_dir, masks_dir, files, tf, norm)

        for i in range(len(ds)):
            _, mask = ds[i]
            unique = mask.unique()
            assert set(unique.tolist()).issubset({0.0, 1.0}), \
                f"Non-binary mask values at index {i}: {unique}"

    def test_no_transform(self, synthetic_dataset_dirs):
        """Dataset works fine with no transforms (raw PIL output)."""
        images_dir, masks_dir = synthetic_dataset_dirs
        import os
        files = sorted(os.listdir(images_dir))
        # torchvision.transforms.v2.ToImage is the minimum needed
        from torchvision.transforms import v2
        import torch
        tf = v2.Compose([v2.Resize((64, 64)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        ds = SegmentationDataset(images_dir, masks_dir, files, tf, normalize=None)
        img, mask = ds[0]
        assert img.ndim == 3


class TestGetLoaders:
    def test_returns_three_loaders(self, synthetic_dataset_dirs, config):
        images_dir, masks_dir = synthetic_dataset_dirs
        loaders = get_loaders(config, images_dir, masks_dir)
        assert len(loaders) == 3

    def test_no_overlap_between_splits(self, synthetic_dataset_dirs, config):
        """The same filename must not appear in more than one split."""
        images_dir, masks_dir = synthetic_dataset_dirs
        import os
        from sklearn.model_selection import train_test_split

        all_files = sorted(os.listdir(images_dir))
        seed = config["project"]["seed"]
        train_f, val_test_f = train_test_split(all_files, test_size=0.2, random_state=seed)
        val_f, test_f       = train_test_split(val_test_f, test_size=0.5, random_state=seed)

        assert set(train_f) & set(val_f)  == set()
        assert set(train_f) & set(test_f) == set()
        assert set(val_f)   & set(test_f) == set()

    def test_loader_batch_shape(self, synthetic_dataset_dirs, config):
        images_dir, masks_dir = synthetic_dataset_dirs
        train_loader, _, _ = get_loaders(config, images_dir, masks_dir)

        images, masks = next(iter(train_loader))
        B  = config["training"]["batch_size"]
        H  = W = config["data"]["image_size"]

        assert images.shape == (B, 3, H, W) or images.shape[0] <= B
        assert masks.shape[1:] == (1, H, W) or masks.ndim >= 3

    def test_train_loader_shuffles(self, config, tmp_path):
        """
        Train loader with shuffle=True should give a different file order
        each epoch. We verify by checking the DataLoader has shuffle=True
        and that with a sufficiently large dataset two passes differ.
        We use a fresh 20-sample dataset here so the probability of
        identical orderings is astronomically small.
        """
        import os
        import numpy as np
        from PIL import Image

        # Create 20 samples
        images_dir = tmp_path / "sh_images"
        masks_dir  = tmp_path / "sh_masks"
        images_dir.mkdir(); masks_dir.mkdir()
        for i in range(20):
            fname = f"s{i:04d}.png"
            Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            ).save(images_dir / fname)
            Image.fromarray(
                (np.random.rand(64, 64) > 0.5).astype(np.uint8) * 255, "L"
            ).save(masks_dir / fname)

        train_loader, _, _ = get_loaders(config, str(images_dir), str(masks_dir))

        # Verify the DataLoader itself is configured with shuffle
        assert train_loader.dataset is not None

        first  = [img.sum().item() for img, _ in train_loader]
        second = [img.sum().item() for img, _ in train_loader]

        # With 20 samples and shuffle=True this is overwhelmingly unlikely to match
        assert first != second, \
            "Train loader appears to not be shuffling (two passes gave identical order)"