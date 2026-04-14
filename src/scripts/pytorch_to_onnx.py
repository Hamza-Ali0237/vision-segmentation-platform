"""
Export a trained PyTorch checkpoint to ONNX format.

Usage:
    python pytorch_to_onnx.py \
        --checkpoint path/to/checkpoint.pth \
        --arch unet \
        --config training/configs/base.yaml \
        --output model.onnx \
        --image-size 256
"""

import argparse
import torch
import yaml
from pathlib import Path

from src.scripts.protocols import build_model


def export_to_onnx(checkpoint_path, arch, config, output_path, image_size=256, opset_version=17):
    device = torch.device("cpu")   # export on CPU for portability

    model = build_model(arch, config)
    state = torch.load(checkpoint_path, map_location=device)

    # Support both raw state_dict and dicts that wrap it
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    dummy_input = torch.randn(1, config["data"]["in_channels"], image_size, image_size)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size", 2: "height", 3: "width"},
            "logits": {0: "batch_size", 2: "height", 3: "width"},
        },
        do_constant_folding=True,
    )

    print(f"ONNX model saved to: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--arch", required=True, choices=["unet", "segnet"])
    parser.add_argument("--config", default="training/configs/base.yaml")
    parser.add_argument("--output", default="model.onnx")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        arch=args.arch,
        config=config,
        output_path=args.output,
        image_size=args.image_size,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()