import torchmetrics

def get_metrics(num_classes: int, device):

    metrics = torchmetrics.MetricCollection({
        'dice_score': torchmetrics.segmentation.DiceScore(
            num_classes, include_background=True, average="micro", input_format="one_hot"
        ),
        "miou": torchmetrics.segmentation.MeanIoU(
            num_classes, include_background=True, per_class=False, input_format="one_hot"
        )
    }).to(device)

    return metrics