"""
Launch a SageMaker Training Job for the CXR Segmentation Platform.

Before running:
  1. Fill in base.yaml  aws.bucket, aws.role_arn, aws.ecr_image_uri
  2. pip install sagemaker boto3
  3. aws configure  (set your access key + region)
  4. Push your Docker image to ECR  (see docker/Dockerfile header)
  5. Upload your dataset to S3:
       aws s3 sync data/images s3://YOUR_BUCKET/data/images/
       aws s3 sync data/masks  s3://YOUR_BUCKET/data/masks/

Usage:
    python aws/train_job.py --config training/configs/base.yaml --arch unet
"""

import argparse
import os
import yaml
import boto3
import sagemaker
from sagemaker.estimator import Estimator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/configs/base.yaml")
    parser.add_argument("--arch",   default="unet", choices=["unet", "segnet", "all"])
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    aws_cfg = config["aws"]
    bucket = aws_cfg["bucket"]
    region = aws_cfg["region"]
    role = aws_cfg["role_arn"]
    image = aws_cfg["ecr_image_uri"]

    session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    job_name = f"cxr-seg-{args.arch}-train"

    # S3 paths
    s3_data_uri = f"s3://{bucket}/data"
    s3_output_uri = f"s3://{bucket}/sagemaker-output/{job_name}"

    # Hyperparameters passed as CLI args to train.py inside the container
    hyperparameters = {
        "config": "training/configs/base.yaml",
        "arch": args.arch,
        "images-dir": "/opt/ml/input/data/training/images",
        "masks-dir": "/opt/ml/input/data/training/masks",
    }
    if args.epochs:
        hyperparameters["epochs"] = args.epochs
    if config["training"].get("batch_size"):
        hyperparameters["batch-size"] = config["training"]["batch_size"]

    estimator = Estimator(
        image_uri=image,
        role=role,
        instance_type=aws_cfg.get("instance_type", "ml.g4dn.xlarge"),
        instance_count=aws_cfg.get("instance_count", 1),
        output_path=s3_output_uri,
        hyperparameters=hyperparameters,
        base_job_name=job_name,
        sagemaker_session=session,
        # Keep training output + model artifacts for 7 days
        volume_size=30,      # GB of EBS attached to the instance
        max_run=4 * 3600,    # 4 hour wall-clock limit — protects your credits
        environment={
            "MODEL_ARCH": args.arch,
            "IMAGE_SIZE": str(config["data"]["image_size"]),
            "PYTHONPATH": "/opt/ml/code",
            # Tell MLflow to upload artifacts to S3
            "MLFLOW_S3_ENDPOINT_URL": "",  # leave blank for default AWS S3
        },
    )

    print(f"\nStarting SageMaker training job: {job_name}")
    print(f"  Instance : {aws_cfg.get('instance_type', 'ml.g4dn.xlarge')}")
    print(f"  Data     : {s3_data_uri}")
    print(f"  Output   : {s3_output_uri}")
    print(f"  Arch     : {args.arch}\n")

    estimator.fit(
        inputs={"training": s3_data_uri},
        job_name=job_name,
        wait=True,   # blocks until job finishes — set False to run async
        logs=True,
    )

    print("\nTraining complete!")
    print(f"Model artifacts at: {estimator.model_data}")

    # Save the model S3 path so deploy.py can pick it up
    output_info = {
        "model_data": estimator.model_data,
        "job_name": job_name,
        "arch": args.arch,
        "image_uri": image,
        "role": role,
    }
    import json
    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/last_training_job_{args.arch}.json", "w") as f:
        json.dump(output_info, f, indent=2)
    print(f"Job info saved to outputs/last_training_job_{args.arch}.json")


if __name__ == "__main__":
    main()