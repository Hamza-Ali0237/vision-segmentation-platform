import os
import io
import json
import base64
import logging
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import v2

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constants
IMAGE_SIZE  = int(os.environ.get("IMAGE_SIZE", 256))
THRESHOLD   = float(os.environ.get("PRED_THRESHOLD", 0.5))
ARCH        = os.environ.get("MODEL_ARCH", "unet").lower()

_NORMALIZE = v2.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
_RESIZE = v2.Compose([
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])


# model_fn
def model_fn(model_dir: str):
    import sys
    sys.path.insert(0, "/opt/ml/code")   # SageMaker copies source_dir here

    from src.scripts.protocols import build_model

    # Look for arch-specific checkpoint first, then any .pth
    ckpt_candidates = [
        os.path.join(model_dir, f"best_{ARCH}.pth"),
        os.path.join(model_dir, "best_unet.pth"),
        os.path.join(model_dir, "best_segnet.pth"),
    ]
    ckpt_path = next((p for p in ckpt_candidates if os.path.exists(p)), None)
    if ckpt_path is None:
        # Fall back to first .pth file found
        pths = list(Path(model_dir).glob("*.pth"))
        if not pths:
            raise FileNotFoundError(f"No .pth checkpoint found in {model_dir}")
        ckpt_path = str(pths[0])

    logger.info(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    saved_config = state.get("config", {})

    # Minimal config fallback
    config = {
        "model": {"num_classes": 1},
        "data": {"in_channels": 3},
    }
    if saved_config:
        config = saved_config

    arch = state.get("arch", ARCH)
    model = build_model(arch, config)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    logger.info(f"Model loaded ({arch}) on {device}")
    return model


# input_fn
def input_fn(request_body, content_type: str):
    if content_type in ("application/octet-stream", "image/png", "image/jpeg"):
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
    elif content_type == "application/json":
        body = json.loads(request_body)
        img_bytes = base64.b64decode(body["image"])
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

    return image


# predict_fn
def predict_fn(image: Image.Image, model: torch.nn.Module):
    device = next(model.parameters()).device

    tensor = _RESIZE(image) # (3, H, W) float32
    tensor = _NORMALIZE(tensor) # normalise
    tensor = tensor.unsqueeze(0).to(device) # (1, 3, H, W)

    with torch.no_grad():
        logits = model(tensor) # (1, 1, H, W)
        probs = torch.sigmoid(logits)
        mask = (probs > THRESHOLD).long()

    return {
        "probs": probs.squeeze().cpu().numpy(),
        "mask": mask.squeeze().cpu().numpy(),
    }


# output_fn
def output_fn(prediction: dict, accept: str):
    response = {
        "mask":  prediction["mask"].tolist(),
        "probs": prediction["probs"].tolist(),
    }
    return json.dumps(response), "application/json"