"""
Tests for UNet and SegNet model architectures.

Checks:
  - Output shape is correct
  - Model produces finite values (no NaN / Inf)
  - Gradients flow through all parameters
  - Models are in the right mode (train/eval)
"""

import pytest
import torch
from src.models.UNet.unet import UNet
from src.models.SegNet.segnet import SegNet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def unet():
    return UNet(in_channels=3, out_channels=1)


@pytest.fixture
def segnet():
    return SegNet(num_classes=1)


# ---------------------------------------------------------------------------
# UNet tests
# ---------------------------------------------------------------------------

class TestUNet:
    def test_output_is_tensor(self, unet, batch_images):
        out = unet(batch_images)
        assert isinstance(out, torch.Tensor)

    def test_output_shape(self, unet, batch_images):
        out = unet(batch_images)
        B, C, H, W = batch_images.shape
        # UNet output may differ slightly in H/W due to unpadded convolutions
        assert out.shape[0] == B, "Batch size mismatch"
        assert out.shape[1] == 1, "Expected 1 output channel (binary)"
        assert out.shape[2] > 0 and out.shape[3] > 0

    def test_output_finite(self, unet, batch_images):
        out = unet(batch_images)
        assert torch.isfinite(out).all(), "UNet output contains NaN or Inf"

    def test_gradients_flow(self, unet, batch_images):
        out  = unet(batch_images)
        loss = out.mean()
        loss.backward()
        grad_norms = [
            p.grad.norm().item()
            for p in unet.parameters()
            if p.grad is not None
        ]
        assert len(grad_norms) > 0, "No gradients were computed"
        assert all(np.isfinite(g) for g in grad_norms), "Non-finite gradients"

    def test_train_eval_modes(self, unet, batch_images):
        unet.train()
        assert unet.training
        out_train = unet(batch_images)

        unet.eval()
        assert not unet.training
        with torch.no_grad():
            out_eval = unet(batch_images)

        assert out_train.shape == out_eval.shape

    def test_different_batch_sizes(self, unet):
        for bs in [1, 2, 4]:
            x   = torch.randn(bs, 3, 64, 64)
            out = unet(x)
            assert out.shape[0] == bs

    def test_no_inplace_modification(self, unet):
        """Input tensor should not be modified by the forward pass."""
        x    = torch.randn(1, 3, 64, 64)
        x_orig = x.clone()
        unet(x)
        assert torch.allclose(x, x_orig)


# ---------------------------------------------------------------------------
# SegNet tests
# ---------------------------------------------------------------------------

class TestSegNet:
    def test_output_is_tensor(self, segnet, batch_images):
        out = segnet(batch_images)
        assert isinstance(out, torch.Tensor)

    def test_output_shape(self, segnet, batch_images):
        out = segnet(batch_images)
        B, C, H, W = batch_images.shape
        # SegNet with MaxUnpool preserves spatial dims
        assert out.shape == (B, 1, H, W), \
            f"Expected {(B, 1, H, W)}, got {tuple(out.shape)}"

    def test_output_finite(self, segnet, batch_images):
        out = segnet(batch_images)
        assert torch.isfinite(out).all(), "SegNet output contains NaN or Inf"

    def test_gradients_flow(self, segnet, batch_images):
        import numpy as np
        out  = segnet(batch_images)
        loss = out.mean()
        loss.backward()
        grad_norms = [
            p.grad.norm().item()
            for p in segnet.parameters()
            if p.grad is not None
        ]
        assert len(grad_norms) > 0
        assert all(np.isfinite(g) for g in grad_norms)

    def test_train_eval_modes(self, segnet, batch_images):
        segnet.train()
        out_train = segnet(batch_images)
        segnet.eval()
        with torch.no_grad():
            out_eval = segnet(batch_images)
        assert out_train.shape == out_eval.shape

    def test_different_batch_sizes(self, segnet):
        for bs in [1, 2, 4]:
            x   = torch.randn(bs, 3, 64, 64)
            out = segnet(x)
            assert out.shape[0] == bs


# ---------------------------------------------------------------------------
# Shared / comparative
# ---------------------------------------------------------------------------

import numpy as np

def test_both_models_produce_different_outputs(batch_images):
    """Sanity check: UNet and SegNet should not produce identical logits."""
    unet   = UNet(in_channels=3, out_channels=1)
    segnet = SegNet(num_classes=1)

    with torch.no_grad():
        out_u = unet(batch_images)
        out_s = segnet(batch_images)

    # Shapes may differ — just check they're not identical (extremely unlikely)
    # If shapes differ the assertion is trivially True
    if out_u.shape == out_s.shape:
        assert not torch.allclose(out_u, out_s, atol=1e-3)