import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset_class import SegmentationDataset
from augmentation import get_transforms

import os

def get_loader(config, images_dir, masks_dir):

    all_images = set(os.listdir(images_dir))
    all_masks = set(os.listdir(masks_dir))
    paired_filenames = sorted(list(all_images & all_masks))

    # Splitting the filenames before creating the datasets
    train_files, val_files = train_test_split(
        paired_filenames,
        test_size=0.2,
        random_state=config["project"]["seed"]
    )

    train_transforms, normalize = get_transforms(config, is_train=True)
    val_transforms, _ = get_transforms(config, is_train=False)

    train_dataset = SegmentationDataset(images_dir, masks_dir, train_files, train_transforms, normalize)
    val_dataset = SegmentationDataset(images_dir, masks_dir, val_files, val_transforms, normalize)

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

    return train_loader, val_loader