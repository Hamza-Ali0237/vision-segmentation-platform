import torch
from torch.utils.data import Dataset

import os
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        
        # Filter out unpaired images and masks
        all_images = set(os.listdir(images_dir))
        all_masks = set(os.listdir(masks_dir))
        paired_files = sorted(all_images & all_masks)
        self.image_filenames = paired_files

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, index):
        image_dir = os.path.join(self.images_dir, self.image_filenames[index])
        mask_dir = os.path.join(self.masks_dir, self.image_filenames[index])

        image = Image.open(image_dir).convert("RGB")
        mask = Image.open(mask_dir).convert("L")

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = mask.squeeze(0)
        mask = torch.where(mask > 0, 1, 0).float()

        return image, mask
