"""
Tests for the FastAPI application.

Uses FastAPI's TestClient so no real HTTP server or SageMaker endpoint
is needed. The SageMaker client is mocked out entirely.
"""

import base64
import io
import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image


# ---------------------------------------------------------------------------
# Helper — make a tiny PNG as bytes and as base64
# ---------------------------------------------------------------------------

def make_png_bytes(w: int = 32, h: int = 32) -> bytes:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def make_png_b64(w: int = 32, h: int = 32) -> str:
    return base64.b64encode(make_png_bytes(w, h)).decode()


# ---------------------------------------------------------------------------
# Mock SageMaker response
# ---------------------------------------------------------------------------

MOCK_MASK  = np.zeros((32, 32), dtype=np.int32)
MOCK_MASK[10:20, 10:20] = 1   # a square of foreground pixels
MOCK_PROBS = MOCK_MASK.astype(np.float32) * 0.9 + 0.05


def mock_predict(image_bytes, arch="unet", content_type="application/octet-stream"):
    return {"mask": MOCK_MASK.copy(), "probs": MOCK_PROBS.copy()}


def mock_list_endpoints():
    return ["unet", "segnet"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """
    TestClient with SageMakerClient fully mocked so tests don't need
    real AWS credentials or a live endpoint.
    """
    import api.main  # ensure module is loaded before patch targets it

    with patch("api.main.sm_client") as mock_sm:
        mock_sm.predict.side_effect = mock_predict
        mock_sm.list_available_endpoints.return_value = ["unet", "segnet"]
        mock_sm.get_endpoint_name.return_value = "cxr-seg-unet-endpoint"

        from api.main import app
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestRoot:
    def test_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_contains_docs_link(self, client):
        body = resp = client.get("/").json()
        assert "docs" in body


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_ok(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_lists_endpoints(self, client):
        body = client.get("/health").json()
        assert "unet"   in body["endpoints_available"]
        assert "segnet" in body["endpoints_available"]


# ---------------------------------------------------------------------------
# POST /predict  (base64)
# ---------------------------------------------------------------------------

class TestPredictBase64:
    def test_returns_200(self, client):
        payload = {"image": make_png_b64(), "arch": "unet"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client):
        payload = {"image": make_png_b64(), "arch": "unet"}
        body = client.post("/predict", json=payload).json()
        for field in ("arch", "mask", "probs", "foreground_ratio", "image_size"):
            assert field in body, f"Missing field: {field}"

    def test_arch_echoed_in_response(self, client):
        payload = {"image": make_png_b64(), "arch": "segnet"}
        body = client.post("/predict", json=payload).json()
        assert body["arch"] == "segnet"

    def test_mask_is_2d_binary(self, client):
        payload = {"image": make_png_b64(), "arch": "unet"}
        body = client.post("/predict", json=payload).json()
        mask = np.array(body["mask"])
        assert mask.ndim == 2
        assert set(np.unique(mask).tolist()).issubset({0, 1})

    def test_probs_in_valid_range(self, client):
        payload = {"image": make_png_b64(), "arch": "unet"}
        body = client.post("/predict", json=payload).json()
        probs = np.array(body["probs"])
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_foreground_ratio_in_range(self, client):
        payload = {"image": make_png_b64(), "arch": "unet"}
        body = client.post("/predict", json=payload).json()
        assert 0.0 <= body["foreground_ratio"] <= 1.0

    def test_image_size_is_list_of_two(self, client):
        payload = {"image": make_png_b64(), "arch": "unet"}
        body = client.post("/predict", json=payload).json()
        assert len(body["image_size"]) == 2

    def test_invalid_base64_returns_422(self, client):
        payload = {"image": "not-valid-base64!!!", "arch": "unet"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_invalid_arch_returns_400(self, client):
        with patch("api.main.sm_client.predict",
                   side_effect=ValueError("Unknown architecture 'resnet'")):
            payload = {"image": make_png_b64(), "arch": "resnet"}
            resp = client.post("/predict", json=payload)
            assert resp.status_code == 400

    def test_sagemaker_down_returns_503(self, client):
        with patch("api.main.sm_client.predict",
                   side_effect=RuntimeError("Endpoint not found")):
            payload = {"image": make_png_b64(), "arch": "unet"}
            resp = client.post("/predict", json=payload)
            assert resp.status_code == 503

    def test_custom_threshold_applied(self, client):
        """Threshold=1.0 should produce an all-zero mask."""
        payload = {"image": make_png_b64(), "arch": "unet", "threshold": 1.0}
        body = client.post("/predict", json=payload).json()
        mask = np.array(body["mask"])
        assert mask.sum() == 0, "All pixels should be 0 with threshold=1.0"

    def test_threshold_out_of_range_returns_422(self, client):
        payload = {"image": make_png_b64(), "arch": "unet", "threshold": 1.5}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /predict/file  (multipart upload)
# ---------------------------------------------------------------------------

class TestPredictFile:
    def test_returns_200(self, client):
        png = make_png_bytes()
        resp = client.post(
            "/predict/file",
            files={"file": ("xray.png", png, "image/png")},
            params={"arch": "unet"},
        )
        assert resp.status_code == 200

    def test_response_fields_present(self, client):
        png = make_png_bytes()
        body = client.post(
            "/predict/file",
            files={"file": ("xray.png", png, "image/png")},
            params={"arch": "unet"},
        ).json()
        for field in ("arch", "mask", "probs", "foreground_ratio", "image_size"):
            assert field in body

    def test_unsupported_file_type_returns_415(self, client):
        resp = client.post(
            "/predict/file",
            files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
            params={"arch": "unet"},
        )
        assert resp.status_code == 415

    def test_jpeg_accepted(self, client):
        # Create a JPEG in memory
        buf = io.BytesIO()
        Image.fromarray(
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        ).save(buf, format="JPEG")
        buf.seek(0)

        resp = client.post(
            "/predict/file",
            files={"file": ("xray.jpg", buf.read(), "image/jpeg")},
            params={"arch": "segnet"},
        )
        assert resp.status_code == 200