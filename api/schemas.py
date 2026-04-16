"""
Pydantic schemas for request validation and response serialisation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    """
    Body for POST /predict when sending base64-encoded images.
    The image must be base64-encoded PNG or JPEG.
    """
    image: str = Field(..., description="Base64-encoded PNG or JPEG image")
    arch: Optional[str] = Field(
        default="unet",
        description="Model architecture to use: 'unet' or 'segnet'",
    )
    threshold: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Sigmoid threshold for binary mask (0.0 – 1.0)",
    )


class PredictResponse(BaseModel):
    """Response from POST /predict"""
    arch: str = Field(..., description="Architecture used for this prediction")
    mask: list = Field(..., description="2-D binary mask (0/1 integers)")
    probs: list = Field(..., description="2-D probability map (0.0 – 1.0 floats)")
    foreground_ratio: float = Field(
        ..., description="Fraction of pixels predicted as foreground"
    )
    image_size: list = Field(..., description="[height, width] of the output mask")


class HealthResponse(BaseModel):
    status: str
    endpoints_available: list[str]
    model: str = "vision-segmentation-platform"