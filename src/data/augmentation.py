from torchvision.transforms import v2
import torch

def get_transforms(config):
    img_size = config['data']["image_size"]
    rot_limit = config['augmentation']['rotate_limit']
    apply_aug = config['augmentation']['apply']

    # Base transforms required for both Train and Validation
    base_transforms = [
        v2.Resize((img_size, img_size)),
        v2.ToImage(), # Modern way to handle tensors
        v2.ToDtype(torch.float32, scale=True)
    ]

    if apply_aug:
        # Insert spatial augmentations BEFORE converting to tensor
        spatial_aug = [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=rot_limit)
        ]

        train_transforms = v2.Compose(spatial_aug + base_transforms)
    else:
        train_transforms = v2.Compose(base_transforms)
    
    # Normalization (applies only to the image)
    normalize = v2.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )

    return train_transforms, normalize

