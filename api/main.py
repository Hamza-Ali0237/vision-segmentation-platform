"""
FastAPI application for the Vision Segmentation Platform.

Architecture:
    Client  →  FastAPI (this file, runs on EC2)  →  SageMaker Endpoint

Endpoints:
    GET  /              — welcome message
    GET  /health        — liveness check + lists deployed endpoints
    POST /predict       — base64 image body → binary mask + probs
    POST /predict/file  — multipart file upload → binary mask + probs

Run locally (for dev/testing without a real SageMaker endpoint):
    uvicorn api.main:app --reload --port 8000

Run in production (on EC2):
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import io
import os
import base64
import logging
import numpy as np

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import PredictRequest, PredictResponse, HealthResponse
from api.sagemaker_client import SageMakerClient

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Vision Segmentation Platform API",
    description=(
        "REST API for chest X-ray segmentation using UNet and SegNet. "
        "Models are served via AWS SageMaker endpoints."
    ),
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
)

# Allow all origins for now — tighten this in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# SageMaker client (singleton)
# ---------------------------------------------------------------------------
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
sm_client  = SageMakerClient(region=AWS_REGION)

# Allow endpoint name overrides via env vars
if ep := os.environ.get("ENDPOINT_UNET"):
    sm_client.set_endpoint("unet", ep)
if ep := os.environ.get("ENDPOINT_SEGNET"):
    sm_client.set_endpoint("segnet", ep)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_response(result: dict, arch: str) -> PredictResponse:
    mask  = result["mask"]
    probs = result["probs"]
    h, w  = mask.shape

    foreground_ratio = float(mask.sum()) / (h * w)

    return PredictResponse(
        arch=arch,
        mask=mask.tolist(),
        probs=probs.tolist(),
        foreground_ratio=round(foreground_ratio, 4),
        image_size=[h, w],
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["General"])
def root():
    return {
        "message": "Vision Segmentation Platform API",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    """
    Liveness check. Also reports which SageMaker endpoints are currently
    deployed and in service.
    """
    available = sm_client.list_available_endpoints()
    return HealthResponse(
        status="ok",
        endpoints_available=available,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_base64(request: PredictRequest):
    """
    Run segmentation on a base64-encoded image.

    - **image**: base64-encoded PNG or JPEG string
    - **arch**: `unet` (default) or `segnet`
    - **threshold**: sigmoid threshold for binary mask (default 0.5)

    Returns the binary mask, probability map, foreground ratio,
    and output image size.
    """
    # Decode base64 → raw bytes
    try:
        image_bytes = base64.b64decode(request.image)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Invalid base64 string. Encode your image with base64.b64encode().",
        )

    logger.info(f"Predict request | arch={request.arch} | threshold={request.threshold}")

    try:
        result = sm_client.predict(image_bytes, arch=request.arch)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Apply custom threshold if different from the default 0.5
    # (the SageMaker handler applies 0.5 — we re-threshold here if needed)
    if request.threshold != 0.5:
        result["mask"] = (result["probs"] > request.threshold).astype(np.int32)

    return _build_response(result, arch=request.arch)


@app.post("/predict/file", response_model=PredictResponse, tags=["Prediction"])
async def predict_file(
    file: UploadFile = File(..., description="PNG or JPEG image file"),
    arch: str = Query(default="unet", description="Model architecture: unet or segnet"),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0),
):
    """
    Run segmentation on an uploaded image file (multipart/form-data).

    Easier to use from a browser or Swagger UI than the base64 endpoint.

    - **file**: PNG or JPEG image upload
    - **arch**: `unet` (default) or `segnet`
    - **threshold**: sigmoid threshold for binary mask (default 0.5)
    """
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Use PNG or JPEG.",
        )

    image_bytes = await file.read()
    logger.info(
        f"Predict file request | arch={arch} | "
        f"filename={file.filename} | size={len(image_bytes)} bytes"
    )

    try:
        result = sm_client.predict(image_bytes, arch=arch)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if threshold != 0.5:
        result["mask"] = (result["probs"] > threshold).astype(np.int32)

    return _build_response(result, arch=arch)


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs."},
    )