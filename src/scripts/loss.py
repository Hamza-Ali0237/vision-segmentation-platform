import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
    def __init__(self, weight_bce=1.0, weight_dice=1.0, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # Compute BCE Loss (Numerically stable)
        bce = self.bce_loss(inputs, targets)

        # Apply sigmoid to get probabilities for Dice
        inputs = torch.sigmoid(inputs)

        # Flatten the tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Compute Dice Loss
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        # Combine BCE and Dice Loss with weights
        total_loss = (self.weight_bce * bce) + (self.weight_dice * dice_loss)
        return total_loss