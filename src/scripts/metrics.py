import torch
import torchmetrics
from torchmetrics.segmentation import DiceScore, MeanIoU

def get_metrics(num_classes: int, device: torch.device) -> torchmetrics.MetricCollection:

    metrics = torchmetrics.MetricCollection({
        'dice_score': DiceScore(
            num_classes, include_background=True, average="micro", input_format="one-hot"
        ),
        "miou": MeanIoU(
            num_classes, include_background=True, per_class=False, input_format="one-hot"
        )
    }).to(device)

    return metrics

def logits_to_one_hot(logits: torch.Tensor, num_classes: int) -> tuple:
    probs = torch.sigmoid(logits)          # (B, 1, H, W)
    pred_binary = (probs > 0.5).long()     # (B, 1, H, W)  0 or 1
 
    if num_classes == 1:
        # torchmetrics one_hot for binary: stack [1-p, p] on dim=1 → (B, 2, H, W)
        # but DiceScore with num_classes=1 wants (B, 1, H, W) — keep as-is
        return pred_binary, logits.new_zeros(logits.shape).long()
 
    # Multi-class path (not used here but kept for extensibility)
    raise NotImplementedError("Multi-class one-hot conversion not implemented yet.")