import torch
import torchmetrics


def validate_one_epoch(model, dataloader, criterion, metrics, device):
    model.eval()
    metrics.reset()
    running_loss = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1) # (B, 1, H, W)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).long()
            targets = masks.long()
            metrics.update(preds, targets)

    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss, metrics.compute()