import torch
import torch.nn as nn
import torch.optim as optim

def training_setup(model, lr):
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10
    )

    return criterion, optimizer, scheduler