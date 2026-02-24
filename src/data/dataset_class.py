import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import tv_tensors

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, filenames, spatial_transform=None, normalize=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        
        self.spatial_transform = spatial_transform
        self.normalize = normalize
        
        self.image_filenames = filenames

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.images_dir, self.image_filenames[index])
        mask_path = os.path.join(self.masks_dir, self.image_filenames[index])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Wrapping image and mask so v2 transforms know how to treat them
        image = tv_tensors.Image(image)
        mask = tv_tensors.Mask(mask)

        # Applying spatial transforms to both simultaneously
        if self.spatial_transform:
            image, mask = self.spatial_transform(image, mask)

        # Apply color normalization to the image only
        if self.normalize:
            image = self.normalize(image)

        # Ensure strict binary format (0.0 or 1.0) and float32 dtype for the loss function
        mask = (mask > 0).to(torch.float32)
        
        # mask = mask.squeeze(0) # [Batch, Height, Width]

        return image, mask