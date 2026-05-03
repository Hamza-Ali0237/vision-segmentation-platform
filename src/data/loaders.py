import os

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.data.dataset_class import SegmentationDataset
from src.data.augmentation import get_transforms

def get_loaders(config: dict, images_dir: str, masks_dir: str):

    all_images = set(os.listdir(images_dir))
    all_masks = set(os.listdir(masks_dir))
    paired_filenames = sorted(list(all_images & all_masks))

    # Splitting the filenames before creating the datasets
    train_files, val_test_files = train_test_split(
        paired_filenames,
        test_size=0.2,
        random_state=config["project"]["seed"]
    )

    val_files, test_files = train_test_split(
        val_test_files,
        test_size=0.5,
        random_state=config["project"]["seed"]
    )

    train_transforms, normalize = get_transforms(config, is_train=True)
    val_test_transforms, _ = get_transforms(config, is_train=False)

    train_dataset = SegmentationDataset(images_dir, masks_dir, train_files, train_transforms, normalize)
    val_dataset = SegmentationDataset(images_dir, masks_dir, val_files, val_test_transforms, normalize)
    test_dataset = SegmentationDataset(images_dir, masks_dir, test_files, val_test_transforms, normalize)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config['data']['num_workers'],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config['data']['num_workers'],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config['data']['num_workers'],
    )

    return train_loader, val_loader, test_loader