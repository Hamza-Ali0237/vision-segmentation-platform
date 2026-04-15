"""
Tests for the metrics setup (DiceScore + MeanIoU via torchmetrics).
"""

import pytest
import torch
from src.scripts.metrics import get_metrics


@pytest.fixture
def metrics(device):
    return get_metrics(num_classes=1, device=device)


class TestMetrics:
    def test_metrics_collection_has_expected_keys(self, metrics):
        expected = {"dice_score", "miou"}
        assert expected.issubset(metrics.keys()), \
            f"Missing keys. Got: {set(metrics.keys())}"

    def test_perfect_prediction_high_dice(self, metrics, device):
        """All-correct binary prediction → dice close to 1."""
        preds   = torch.ones(2, 1, 32, 32, dtype=torch.long).to(device)
        targets = torch.ones(2, 1, 32, 32, dtype=torch.long).to(device)

        metrics.reset()
        metrics.update(preds, targets)
        result = metrics.compute()

        assert result["dice_score"] > 0.9, \
            f"Expected dice > 0.9 on perfect preds, got {result['dice_score']:.4f}"

    def test_all_wrong_prediction_low_dice(self, metrics, device):
        """Completely wrong prediction → low dice."""
        preds   = torch.zeros(2, 1, 32, 32, dtype=torch.long).to(device)
        targets = torch.ones(2, 1, 32, 32, dtype=torch.long).to(device)

        metrics.reset()
        metrics.update(preds, targets)
        result = metrics.compute()

        assert result["dice_score"] < 0.5, \
            f"Expected dice < 0.5 on all-wrong preds, got {result['dice_score']:.4f}"

    def test_reset_clears_state(self, metrics, device):
        """After reset, compute should behave as if no updates happened."""
        preds   = torch.ones(2, 1, 32, 32, dtype=torch.long).to(device)
        targets = torch.ones(2, 1, 32, 32, dtype=torch.long).to(device)

        metrics.update(preds, targets)
        metrics.reset()

        # Updating with bad preds after reset should give bad metrics
        bad_preds = torch.zeros(2, 1, 32, 32, dtype=torch.long).to(device)
        metrics.update(bad_preds, targets)
        result = metrics.compute()
        assert result["dice_score"] < 0.9

    def test_metrics_accept_logit_derived_preds(self, metrics, device):
        """Simulate the thresholding done in the train/val loop."""
        logits  = torch.randn(2, 1, 32, 32).to(device)
        targets = (torch.rand(2, 1, 32, 32) > 0.5).long().to(device)

        preds = (torch.sigmoid(logits) > 0.5).long()

        metrics.reset()
        metrics.update(preds, targets)
        result = metrics.compute()

        assert torch.isfinite(result["dice_score"].detach().clone())
        assert torch.isfinite(result["miou"].detach().clone())

    def test_accumulation_across_batches(self, metrics, device):
        """Metrics should accumulate correctly across multiple update calls."""
        metrics.reset()

        for _ in range(3):
            preds   = torch.ones(2, 1, 32, 32, dtype=torch.long).to(device)
            targets = torch.ones(2, 1, 32, 32, dtype=torch.long).to(device)
            metrics.update(preds, targets)

        result = metrics.compute()
        assert result["dice_score"] > 0.9