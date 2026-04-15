import torch
import torchmetrics
from torchmetrics.segmentation import DiceScore, MeanIoU


def get_metrics(num_classes: int, device: torch.device) -> torchmetrics.MetricCollection:
    """
    Returns a MetricCollection with DiceScore and MeanIoU.

    torchmetrics segmentation metrics with input_format="one_hot" expect:
      preds   shape (B, C, H, W)  — thresholded binary, dtype long
      targets shape (B, C, H, W)  — binary ground truth, dtype long

    For binary segmentation (num_classes=1) we handle the conversion in
    the train/val loops before calling metrics.update().

    Compatible with torchmetrics >= 1.3  (import path changed in 1.4+;
    the direct submodule import works across all recent versions).
    """
    metrics = torchmetrics.MetricCollection(
        {
            "dice_score": DiceScore(
                num_classes=num_classes,
                include_background=True,
                average="micro",
                input_format="one-hot",   # hyphen — changed in torchmetrics 1.4+
            ),
            "miou": MeanIoU(
                num_classes=num_classes,
                include_background=True,
                per_class=False,
                input_format="one-hot",
            ),
        }
    ).to(device)

    return metrics


def logits_to_one_hot(logits: torch.Tensor, num_classes: int) -> tuple:
    """
    Convert raw logits + binary masks to one-hot format expected by torchmetrics.

    Returns (preds_one_hot, targets_one_hot), each shape (B, C, H, W).
    For binary (num_classes=1) this just thresholds at 0.5 and stacks.
    """
    probs = torch.sigmoid(logits)          # (B, 1, H, W)
    pred_binary = (probs > 0.5).long()     # (B, 1, H, W)  0 or 1

    if num_classes == 1:
        # torchmetrics one_hot for binary: stack [1-p, p] on dim=1 → (B, 2, H, W)
        # but DiceScore with num_classes=1 wants (B, 1, H, W) — keep as-is
        return pred_binary, logits.new_zeros(logits.shape).long()

    # Multi-class path (not used here but kept for extensibility)
    raise NotImplementedError("Multi-class one-hot conversion not implemented yet.")