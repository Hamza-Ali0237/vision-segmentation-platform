import torch
import torch.nn as nn
import torch.optim as optim
from loss import DiceBCELoss

def training_setup(model, lr=1e-4, weight_decay=1e-4):
    criterion = DiceBCELoss()  

    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5
    )

    return criterion, optimizer, scheduler