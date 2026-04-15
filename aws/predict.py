"""
Send an image to the deployed SageMaker endpoint and print/save the result.

Usage:
    python aws/predict.py --image path/to/xray.png --arch unet
    python aws/predict.py --image path/to/xray.png --endpoint cxr-seg-unet-endpoint
"""

import argparse
import json
import os
import boto3
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input PNG/JPEG")
    parser.add_argument("--arch", default="unet", choices=["unet", "segnet"])
    parser.add_argument("--endpoint", default=None,
                        help="Endpoint name (auto-read from outputs/ if omitted)")
    parser.add_argument("--output", default="outputs/prediction.png",
                        help="Where to save the predicted mask image")
    parser.add_argument("--region", default="us-east-1")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve endpoint name
    endpoint_name = args.endpoint
    if not endpoint_name:
        ep_file = f"outputs/endpoint_{args.arch}.txt"
        if not os.path.exists(ep_file):
            raise FileNotFoundError(
                f"No endpoint file at {ep_file}. Run deploy.py first or pass --endpoint."
            )
        with open(ep_file) as f:
            endpoint_name = f.read().strip()

    # Read image as raw bytes
    with open(args.image, "rb") as f:
        image_bytes = f.read()

    # Call the endpoint
    client = boto3.client("sagemaker-runtime", region_name=args.region)
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/octet-stream",
        Body=image_bytes,
    )

    result = json.loads(response["Body"].read())
    mask = np.array(result["mask"], dtype=np.uint8) * 255   # 0 or 255
    probs = np.array(result["probs"])

    print(f"Predicted mask shape : {mask.shape}")
    print(f"Foreground pixels    : {(mask > 0).sum()} / {mask.size}")
    print(f"Max prob             : {probs.max():.4f}  Min: {probs.min():.4f}")

    # Save mask
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    Image.fromarray(mask).save(args.output)
    print(f"Mask saved to: {args.output}")


if __name__ == "__main__":
    main()