"""
Tests for the SageMaker inference handlers (model_fn / input_fn / predict_fn / output_fn).

These tests do NOT require a running endpoint — they call the handlers directly
just like SageMaker would during serving.
"""

import io
import json
import base64
import pytest
import torch
import numpy as np
from PIL import Image

# Patch sys.path so inference.py can import src.*
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.inference import input_fn, predict_fn, output_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    """Return raw PNG bytes of a random RGB image."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def make_dummy_model(config: dict):
    """Build a tiny SegNet with random weights for testing."""
    from src.scripts.protocols import build_model
    model = build_model("segnet", config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# input_fn
# ---------------------------------------------------------------------------

class TestInputFn:
    def test_octet_stream(self):
        png_bytes = make_png_bytes()
        img = input_fn(png_bytes, "application/octet-stream")
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_image_png(self):
        png_bytes = make_png_bytes()
        img = input_fn(png_bytes, "image/png")
        assert isinstance(img, Image.Image)

    def test_application_json_base64(self):
        png_bytes  = make_png_bytes()
        b64_str    = base64.b64encode(png_bytes).decode("utf-8")
        body       = json.dumps({"image": b64_str}).encode()
        img = input_fn(body, "application/json")
        assert isinstance(img, Image.Image)

    def test_unsupported_content_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported content type"):
            input_fn(b"some bytes", "text/plain")


# ---------------------------------------------------------------------------
# predict_fn
# ---------------------------------------------------------------------------

class TestPredictFn:
    def test_returns_dict_with_expected_keys(self, config):
        model = make_dummy_model(config)
        img   = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        result = predict_fn(img, model)
        assert "mask"  in result
        assert "probs" in result

    def test_mask_is_binary(self, config):
        model = make_dummy_model(config)
        img   = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        result = predict_fn(img, model)
        unique = np.unique(result["mask"])
        assert set(unique.tolist()).issubset({0, 1}), \
            f"Mask contains non-binary values: {unique}"

    def test_probs_in_valid_range(self, config):
        model = make_dummy_model(config)
        img   = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        result = predict_fn(img, model)
        assert result["probs"].min() >= 0.0
        assert result["probs"].max() <= 1.0

    def test_output_shapes_match(self, config):
        model = make_dummy_model(config)
        img   = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        result = predict_fn(img, model)
        assert result["mask"].shape  == result["probs"].shape


# ---------------------------------------------------------------------------
# output_fn
# ---------------------------------------------------------------------------

class TestOutputFn:
    def test_returns_json_string(self):
        prediction = {
            "mask":  np.array([[0, 1], [1, 0]]),
            "probs": np.array([[0.1, 0.9], [0.8, 0.2]]),
        }
        body, content_type = output_fn(prediction, "application/json")
        assert content_type == "application/json"
        parsed = json.loads(body)
        assert "mask"  in parsed
        assert "probs" in parsed

    def test_round_trip(self):
        mask  = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)
        probs = np.array([[0.1, 0.9, 0.2], [0.8, 0.3, 0.7]])
        prediction = {"mask": mask, "probs": probs}

        body, _ = output_fn(prediction, "application/json")
        parsed  = json.loads(body)

        assert parsed["mask"]  == mask.tolist()
        assert parsed["probs"] == probs.tolist()