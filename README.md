# Vision Segmentation Platform

End-to-end binary segmentation of chest X-rays using UNet and SegNet, with MLflow experiment tracking, Optuna hyperparameter search, and AWS SageMaker training + deployment.

---

## Project Structure

```
vision-segmentation-platform/
├── src/
│   ├── data/
│   │   ├── augmentation.py       # torchvision v2 transforms
│   │   ├── dataset_class.py      # SegmentationDataset
│   │   └── loaders.py            # get_loaders() — train/val/test split
│   ├── models/
│   │   ├── UNet/                 # encoder, decoder, bottleneck, unet
│   │   └── SegNet/               # encoder, decoder, segnet
│   └── scripts/
│       ├── loss.py               # DiceBCELoss
│       ├── metrics.py            # DiceScore + MeanIoU via torchmetrics
│       ├── protocols.py          # build_model(), training_setup()
│       ├── train_logic.py        # train_one_epoch()
│       ├── validate.py           # validate_one_epoch()
│       └── visualize.py          # save_predictions() → PNG overlays
├── training/configs/
│   └── base.yaml                 # all hyperparams, paths, AWS config
├── tests/
│   ├── conftest.py               # shared fixtures (config, synthetic data)
│   ├── test_models.py            # UNet + SegNet forward pass tests
│   ├── test_dataset.py           # dataset / loader tests
│   ├── test_loss.py              # DiceBCELoss tests
│   ├── test_metrics.py           # torchmetrics integration tests
│   ├── test_train_step.py        # train/val loop smoke tests
│   ├── test_inference.py         # SageMaker handler tests
│   └── test_api.py               # FastAPI endpoint tests
├── api/
│   ├── main.py                   # FastAPI app — /predict, /predict/file, /health
│   ├── schemas.py                # Pydantic request/response models
│   └── sagemaker_client.py       # boto3 wrapper for SageMaker endpoint calls
├── aws/
│   ├── train_job.py              # launch SageMaker Training Job
│   ├── deploy.py                 # deploy model to SageMaker endpoint
│   ├── predict.py                # call the live endpoint
│   ├── ecr_push.sh               # build + push training Docker image to ECR
│   ├── ecr_push_api.sh           # build + push FastAPI Docker image to ECR
│   └── deploy_api.sh             # deploy FastAPI container to EC2
├── docker/
│   ├── Dockerfile                # SageMaker training + inference container
│   ├── Dockerfile.api            # lightweight FastAPI container
│   └── entrypoint.sh             # routes "train" and "serve" commands
├── inference/
│   ├── inference.py              # model_fn / input_fn / predict_fn / output_fn
│   └── serving.py                # Flask app wrapping inference handlers for /ping and /invocations
├── train.py                      # main training entrypoint
├── hpo.py                        # Optuna HPO (calls train.py per trial)
├── pytorch_to_onnx.py            # export trained checkpoint to ONNX
├── requirements.txt              # training container dependencies
├── requirements-api.txt          # FastAPI container dependencies
└── setup.py
```

---

## Setup

```bash
git clone https://github.com/Hamza-Ali0237/vision-segmentation-platform
cd vision-segmentation-platform

conda create -n vsp python=3.10 -y
conda activate vsp

pip install -e .
pip install -r requirements.txt
pip install "numpy<2"             # required for torch compatibility
```

---

## Local Training

### 1. Configure `training/configs/base.yaml`

The only fields you must fill in before running locally:

```yaml
# Leave paths as-is for local runs — override via CLI flags
paths:
  data_dir: "/opt/ml/input/data/training"   # ignored locally (use --images-dir)
```

### 2. Train a single model

```bash
python train.py \
  --arch unet \
  --images-dir data/images \
  --masks-dir  data/masks
```

Or train both architectures back-to-back:

```bash
python train.py --arch all \
  --images-dir data/images \
  --masks-dir  data/masks
```

### 3. View MLflow runs

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

MLflow logs per-epoch train/val loss, Dice, MIoU, learning rate, and saves checkpoints + visualisation PNGs as artifacts.

After a SageMaker training job, the MLflow DB is packaged inside `model.tar.gz`. Extract it first:

```bash
aws s3 cp s3://your-bucket/sagemaker-output/JOB_NAME/JOB_NAME/output/model.tar.gz ./model.tar.gz
mkdir -p model_extracted && tar -xzf model.tar.gz -C model_extracted
mlflow ui --backend-store-uri sqlite:///model_extracted/mlflow.db
```

---

## Hyperparameter Search (Optuna)

```bash
python hpo.py \
  --arch unet \
  --images-dir data/images \
  --masks-dir  data/masks \
  --n-trials 20
```

Each trial is a full training run logged as a **nested MLflow run** under a parent `hpo-unet` run. The study is persisted to `optuna_unet.db` so you can resume interrupted searches.

Best hyperparams are saved to `outputs/best_params_unet.yaml`.

---

## Tests

```bash
python -m pytest                        # run all tests
python -m pytest tests/test_models.py  # just model shape tests
python -m pytest tests/test_api.py     # just FastAPI tests
python -m pytest -v --tb=long          # verbose with full tracebacks
```

All tests are CPU-only and use synthetic data — no GPU, real dataset, or live AWS endpoint needed.

Install FastAPI test dependencies first if running `test_api.py` locally:

```bash
pip install fastapi httpx python-multipart boto3
```

---

## FastAPI

The FastAPI app sits in front of the SageMaker endpoint and provides a clean REST interface.

### Run locally

```bash
pip install fastapi uvicorn python-multipart boto3
uvicorn api.main:app --reload --port 8000
```

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check, lists deployed endpoints |
| POST | `/predict` | Base64-encoded image in JSON body |
| POST | `/predict/file` | Multipart file upload |

### Example — predict via curl

```bash
# File upload
curl -X POST http://localhost:8000/predict/file \
  -F "file=@path/to/xray.png" \
  -F "arch=unet"

# Base64
IMAGE_B64=$(base64 -i path/to/xray.png)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMAGE_B64\", \"arch\": \"unet\"}"
```

---

## AWS SageMaker Deployment

### Prerequisites

1. AWS account with ~$100 free credits
2. AWS CLI installed and configured (`aws configure`)
3. Docker Desktop installed and running
4. Fill in `training/configs/base.yaml`:
   ```yaml
   aws:
     bucket:        "your-s3-bucket-name"
     role_arn:      "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
     instance_type: "ml.g4dn.xlarge"
     ecr_image_uri: ""   # filled in after Step 2
   ```

### Step 1 — Create IAM Roles

**SageMaker execution role** (AWS Console → IAM → Roles → Create Role):
- Trusted entity: **SageMaker**
- Attach: `AmazonSageMakerFullAccess`, `AmazonS3FullAccess`, `AmazonEC2ContainerRegistryFullAccess`
- Name it `SageMakerExecutionRole` → copy ARN into `base.yaml`

**EC2 role for FastAPI** (for later):
- Trusted entity: **EC2**
- Attach: `AmazonSageMakerFullAccess`, `AmazonEC2ContainerRegistryReadOnly`
- Name it `EC2SageMakerRole`

### Step 2 — Push Docker image to ECR

> **Important:** Always build with `--platform linux/amd64 --provenance=false` to produce a manifest type compatible with SageMaker.

```bash
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin \
    YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

docker build --platform linux/amd64 --provenance=false \
  -f docker/Dockerfile \
  -t YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/cxr-segmentation:latest \
  .

docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/cxr-segmentation:latest
```

Copy the URI into `base.yaml → aws.ecr_image_uri`.

### Step 3 — Upload dataset to S3

```bash
aws s3 mb s3://your-bucket-name --region us-east-1
aws s3 sync data/images s3://your-bucket-name/data/images/
aws s3 sync data/masks  s3://your-bucket-name/data/masks/
```

### Step 4 — Request GPU quota (if needed)

New AWS accounts have a default quota of 0 GPU instances. Request an increase:

AWS Console → Service Quotas → Amazon SageMaker → search `ml.g4dn.xlarge for training job usage` → Request increase to 1.

While waiting, you can use `ml.m5.xlarge` (CPU) in `base.yaml` to verify the pipeline.

### Step 5 — Launch training job

```bash
pip install sagemaker boto3
python aws/train_job.py --arch unet
```

Training logs stream to your terminal. Job info is saved to `outputs/last_training_job_unet.json` when complete.

### Step 6 — Deploy endpoint

```bash
python aws/deploy.py --arch unet
```

### Step 7 — Test the endpoint

```bash
python aws/predict.py --image path/to/xray.png --arch unet
# Mask saved to outputs/prediction.png
```

### Step 8 — Delete the endpoint when done

**Endpoints bill ~$0.74/hr even when idle. Always delete when not in use.**

```bash
aws sagemaker delete-endpoint --endpoint-name cxr-seg-unet-endpoint --region us-east-1
```

---

## Deploy FastAPI to EC2

Once the SageMaker endpoint is live:

```bash
# Build and push FastAPI image
chmod +x aws/ecr_push_api.sh
./aws/ecr_push_api.sh YOUR_ACCOUNT_ID us-east-1

# Create EC2 key pair first (AWS Console → EC2 → Key Pairs)
chmod +x aws/deploy_api.sh
./aws/deploy_api.sh YOUR_ACCOUNT_ID us-east-1 your-key-pair-name
```

API will be available at `http://EC2_PUBLIC_IP:8000`.

---

## Export to ONNX

```bash
python pytorch_to_onnx.py \
  --checkpoint outputs/models/best_unet.pth \
  --arch unet \
  --output outputs/unet.onnx
```

---

## Model Performance

| Model | Test Dice | Test MIoU | Epochs |
|---|---|---|---|
| UNet | 0.9528 | 0.9111 | 14 (early stop) |
| SegNet | 0.9569 | 0.9180 | 15 (1-epoch run) |

Trained on chest X-ray dataset using `ml.g4dn.xlarge` (T4 GPU) on AWS SageMaker.

---

## MLflow Quick Reference

| What | Command |
|---|---|
| View runs locally | `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| View runs from SageMaker job | Extract `mlflow.db` from `model.tar.gz` first |
| Compare two runs | UI → select runs → Compare |
| Download artifact | UI → run → Artifacts tab |
| List experiments | `mlflow experiments list` |

---

## Cost Estimates (AWS Free Credits)

| Resource | Cost | Notes |
|---|---|---|
| `ml.g4dn.xlarge` training | ~$0.74/hr | ~2 hrs per model |
| `ml.g4dn.xlarge` endpoint | ~$0.74/hr | **Delete when done** |
| EC2 t3.micro (FastAPI) | Free tier / ~$0.01/hr | Stop when not in use |
| S3 storage | ~$0.02/GB/month | Minimal |
| ECR storage | ~$0.10/GB/month | ~4GB for both images |

Training both models + a few hours of endpoint testing ≈ **$5–10** total.

---

## Quick Reference

```bash
# Train
python aws/train_job.py --arch unet

# Deploy SageMaker endpoint
python aws/deploy.py --arch unet

# Test endpoint directly
python aws/predict.py --image xray.png --arch unet

# Run FastAPI locally
uvicorn api.main:app --reload --port 8000

# Run tests
python -m pytest

# Delete endpoint when done
aws sagemaker delete-endpoint --endpoint-name cxr-seg-unet-endpoint --region us-east-1
```