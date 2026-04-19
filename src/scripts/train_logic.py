import torch
import torch.nn as nn

def train_one_epoch(model, dataloader, criterion, optimizer, metrics, device):
    model.train()
    metrics.reset()
    running_loss = 0.0
    
    metrics.reset()

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)  # Add channel dim only if missing → (B, 1, H, W)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        
        with torch.no_grad():
            preds = (torch.sigmoid(outputs) > 0.5).long()  # (B,1,H,W)
            targets = masks.long() # (B,1,H,W)
            metrics.update(preds, targets)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, metrics.compute()