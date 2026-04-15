"""
Smoke tests for the full training and validation loops.

These tests verify that:
  - One epoch of training completes without errors
  - Loss decreases over a few steps on a trivially learnable problem
  - Validation loop runs and returns finite metrics
  - Checkpoint saving works
  - Protocols (build_model, training_setup) wire everything up correctly
"""

import os
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.UNet.unet import UNet
from src.models.SegNet.segnet import SegNet
from src.scripts.loss import DiceBCELoss
from src.scripts.metrics import get_metrics
from src.scripts.protocols import build_model, training_setup
from src.scripts.train_logic import train_one_epoch
from src.scripts.validate import validate_one_epoch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tiny_loader(batch_size: int = 2, n_batches: int = 3,
                     img_size: int = 64, device: str = "cpu"):
    """
    Creates a DataLoader backed by random tensors — no disk I/O required.
    Images: (B, 3, H, W), Masks: (B, H, W) binary float.
    """
    imgs  = torch.randn(batch_size * n_batches, 3, img_size, img_size)
    masks = (torch.rand(batch_size * n_batches, img_size, img_size) > 0.5).float()
    ds    = TensorDataset(imgs, masks)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class TestBuildModel:
    def test_build_unet(self, config):
        model = build_model("unet", config)
        assert isinstance(model, UNet)

    def test_build_segnet(self, config):
        model = build_model("segnet", config)
        assert isinstance(model, SegNet)

    def test_unknown_arch_raises(self, config):
        with pytest.raises(ValueError, match="Unknown architecture"):
            build_model("resnet", config)

    def test_training_setup_returns_three(self, config):
        model = build_model("unet", config)
        result = training_setup(model, config)
        assert len(result) == 3   # criterion, optimizer, scheduler


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

class TestTrainOneEpoch:
    @pytest.mark.parametrize("arch", ["unet", "segnet"])
    def test_returns_finite_loss(self, arch, config, device):
        model     = build_model(arch, config).to(device)
        criterion = DiceBCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        metrics   = get_metrics(config["model"]["num_classes"], device)
        loader    = make_tiny_loader()

        loss, metric_vals = train_one_epoch(model, loader, criterion, optimizer, metrics, device)

        assert isinstance(loss, float)
        assert loss > 0
        assert torch.isfinite(torch.tensor(loss))

    @pytest.mark.parametrize("arch", ["unet", "segnet"])
    def test_metrics_keys_present(self, arch, config, device):
        model     = build_model(arch, config).to(device)
        criterion = DiceBCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        metrics   = get_metrics(config["model"]["num_classes"], device)
        loader    = make_tiny_loader()

        _, metric_vals = train_one_epoch(model, loader, criterion, optimizer, metrics, device)

        assert "dice_score" in metric_vals
        assert "miou"        in metric_vals

    def test_loss_decreases_over_steps(self, config, device):
        """
        On a trivially learnable problem (images = masks), loss should
        decrease after enough gradient steps.
        """
        # All-ones mask, all-positive logits → near-zero loss quickly
        imgs  = torch.ones(8, 3, 64, 64)
        masks = torch.ones(8, 64, 64)
        ds    = TensorDataset(imgs, masks)
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        model     = build_model("segnet", config).to(device)
        criterion = DiceBCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        metrics   = get_metrics(config["model"]["num_classes"], device)

        losses = []
        for _ in range(5):
            loss, _ = train_one_epoch(model, loader, criterion, optimizer, metrics, device)
            losses.append(loss)

        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


# ---------------------------------------------------------------------------
# Validate loop
# ---------------------------------------------------------------------------

class TestValidateOneEpoch:
    @pytest.mark.parametrize("arch", ["unet", "segnet"])
    def test_returns_finite_loss(self, arch, config, device):
        model     = build_model(arch, config).to(device)
        model.eval()
        criterion = DiceBCELoss()
        metrics   = get_metrics(config["model"]["num_classes"], device)
        loader    = make_tiny_loader()

        loss, metric_vals = validate_one_epoch(model, loader, criterion, metrics, device)

        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))

    def test_model_stays_in_eval_mode(self, config, device):
        model     = build_model("unet", config).to(device)
        criterion = DiceBCELoss()
        metrics   = get_metrics(config["model"]["num_classes"], device)
        loader    = make_tiny_loader()

        validate_one_epoch(model, loader, criterion, metrics, device)
        assert not model.training, "Model should remain in eval mode after validate_one_epoch"

    def test_no_gradient_computation(self, config, device):
        """Validation must not accumulate gradients."""
        model     = build_model("segnet", config).to(device)
        criterion = DiceBCELoss()
        metrics   = get_metrics(config["model"]["num_classes"], device)
        loader    = make_tiny_loader()

        validate_one_epoch(model, loader, criterion, metrics, device)

        for p in model.parameters():
            assert p.grad is None, "Gradients were computed during validation"


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def test_checkpoint_save_and_load(config, device, tmp_path):
    model = build_model("unet", config).to(device)
    ckpt_path = str(tmp_path / "best_unet.pth")

    state = {
        "epoch":            1,
        "arch":             "unet",
        "model_state_dict": model.state_dict(),
        "val_loss":         0.42,
        "config":           config,
    }
    torch.save(state, ckpt_path)
    assert os.path.exists(ckpt_path)

    # Load back
    loaded = torch.load(ckpt_path, map_location=device)
    model2 = build_model("unet", config).to(device)
    model2.load_state_dict(loaded["model_state_dict"])

    # Both models should produce identical outputs
    x = torch.randn(1, 3, 64, 64).to(device)
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)
    assert torch.allclose(out1, out2)