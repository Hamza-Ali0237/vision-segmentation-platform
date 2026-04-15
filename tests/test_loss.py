"""
Tests for DiceBCELoss.
"""

import pytest
import torch
from src.scripts.loss import DiceBCELoss


@pytest.fixture
def loss_fn():
    return DiceBCELoss(weight_bce=1.0, weight_dice=1.0)


class TestDiceBCELoss:
    def test_returns_scalar(self, loss_fn, batch_images, batch_masks):
        logits = torch.randn_like(batch_masks)
        loss = loss_fn(logits, batch_masks)
        assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"

    def test_loss_is_positive(self, loss_fn, batch_masks):
        logits = torch.randn_like(batch_masks)
        loss = loss_fn(logits, batch_masks)
        assert loss.item() > 0

    def test_loss_is_finite(self, loss_fn, batch_masks):
        logits = torch.randn_like(batch_masks)
        loss = loss_fn(logits, batch_masks)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_gradients_flow(self, loss_fn, batch_masks):
        logits = torch.randn_like(batch_masks, requires_grad=True)
        loss = loss_fn(logits, batch_masks)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_perfect_prediction_lower_than_random(self, loss_fn):
        """Loss on a near-perfect prediction should be lower than on random logits."""
        masks = (torch.rand(2, 1, 64, 64) > 0.5).float()

        # Near-perfect: high positive logits where mask=1, high negative where mask=0
        perfect_logits = masks * 10.0 - (1 - masks) * 10.0
        random_logits  = torch.randn_like(masks)

        perfect_loss = loss_fn(perfect_logits, masks)
        random_loss  = loss_fn(random_logits,  masks)

        assert perfect_loss.item() < random_loss.item(), \
            f"Perfect loss ({perfect_loss:.4f}) not less than random ({random_loss:.4f})"

    def test_all_zeros_prediction(self, loss_fn):
        """Predicting all zeros (no foreground) should still produce a finite loss."""
        masks  = torch.ones(2, 1, 64, 64)
        logits = torch.zeros_like(masks)
        loss = loss_fn(logits, masks)
        assert torch.isfinite(loss)

    def test_weight_scaling(self):
        """Higher weight_dice should increase the dice component's contribution."""
        masks  = (torch.rand(2, 1, 32, 32) > 0.5).float()
        logits = torch.randn_like(masks)

        loss_equal  = DiceBCELoss(weight_bce=1.0, weight_dice=1.0)(logits, masks)
        loss_moredice = DiceBCELoss(weight_bce=1.0, weight_dice=5.0)(logits, masks)

        assert loss_moredice.item() != loss_equal.item(), \
            "Changing weight_dice had no effect on the loss"

    def test_no_nan_with_all_foreground(self, loss_fn):
        """All-foreground mask should not cause NaN via division by zero."""
        masks  = torch.ones(2, 1, 32, 32)
        logits = torch.randn_like(masks)
        loss = loss_fn(logits, masks)
        assert not torch.isnan(loss)

    def test_no_nan_with_all_background(self, loss_fn):
        """All-background mask should not cause NaN."""
        masks  = torch.zeros(2, 1, 32, 32)
        logits = torch.randn_like(masks)
        loss = loss_fn(logits, masks)
        assert not torch.isnan(loss)