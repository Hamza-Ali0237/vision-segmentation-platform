"""
Deploy a trained model from S3 to a SageMaker real-time endpoint.

Reads the job info saved by train_job.py, or accepts explicit args.

Usage:
    # After a training job:
    python aws/deploy.py --arch unet

    # Pointing at an existing model artifact:
    python aws/deploy.py \
        --model-data s3://bucket/sagemaker-output/job/output/model.tar.gz \
        --arch unet
"""

import argparse
import json
import os
import yaml
import boto3
import sagemaker
from sagemaker.model import Model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/configs/base.yaml")
    parser.add_argument("--arch", default="unet", choices=["unet", "segnet"])
    parser.add_argument("--model-data", default=None,
                        help="S3 URI to model.tar.gz (auto-read from last job if omitted)")
    parser.add_argument("--endpoint-name", default=None)
    parser.add_argument("--instance-type", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    aws_cfg = config["aws"]
    region  = aws_cfg["region"]
    role    = aws_cfg["role_arn"]
    image   = aws_cfg["ecr_image_uri"]

    # Resolve model_data
    model_data = args.model_data
    if not model_data:
        job_info_path = f"outputs/last_training_job_{args.arch}.json"
        if not os.path.exists(job_info_path):
            raise FileNotFoundError(
                f"No job info found at {job_info_path}. "
                "Run train_job.py first or pass --model-data explicitly."
            )
        with open(job_info_path) as f:
            job_info = json.load(f)
        model_data = job_info["model_data"]

    endpoint_name = args.endpoint_name or f"cxr-seg-{args.arch}-endpoint"
    instance_type = args.instance_type or aws_cfg.get("instance_type", "ml.g4dn.xlarge")
    session = sagemaker.Session(boto_session=boto3.Session(region_name=region))

    print(f"\nDeploying model to endpoint: {endpoint_name}")
    print(f"  Model data  : {model_data}")
    print(f"  Image       : {image}")
    print(f"  Instance    : {instance_type}\n")

    model = Model(
        model_data=model_data,
        role=role,
        image_uri=image,
        env={
            "MODEL_ARCH": args.arch,
            "IMAGE_SIZE": str(config["data"]["image_size"]),
            "PYTHONPATH": "/opt/ml/code",
        },
        sagemaker_session=session,
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        wait=True,
    )

    print(f"\nEndpoint deployed: {endpoint_name}")
    print("You can now send predictions with aws/predict.py")

    # Save endpoint name for predict.py
    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/endpoint_{args.arch}.txt", "w") as f:
        f.write(endpoint_name)


if __name__ == "__main__":
    main()