import torch
import torch.nn as nn
import torch.optim as optim

from src.models.UNet.unet import UNet
from src.models.SegNet.segnet import SegNet
from src.scripts.loss import DiceBCELoss


def build_model(arch: str, config: dict) -> nn.Module:
    num_classes = config["model"]["num_classes"]
    in_channels = config["data"]["in_channels"]

    arch = arch.lower()
    if arch == "unet":
        return UNet(in_channels=in_channels, out_channels=num_classes)
    elif arch == "segnet":
        return SegNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: '{arch}'. Choose 'unet' or 'segnet'.")


def training_setup(model: nn.Module, config: dict, override: dict = None):
    cfg = config["training"]
    override = override or {}

    lr = override.get("lr", cfg["learning_rate"])
    weight_decay = override.get("weight_decay", cfg["weight_decay"])
    weight_bce = override.get("weight_bce", cfg.get("weight_bce", 1.0))
    weight_dice = override.get("weight_dice", cfg.get("weight_dice", 1.0))
    lr_factor = override.get("lr_factor", cfg["lr_factor"])
    lr_patience = override.get("lr_patience", cfg["lr_patience"])

    criterion = DiceBCELoss(weight_bce=weight_bce, weight_dice=weight_dice)

    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience
    )

    return criterion, optimizer, scheduler