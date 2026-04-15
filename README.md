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
│   └── test_inference.py         # SageMaker handler tests
├── aws/
│   ├── train_job.py              # launch SageMaker Training Job
│   ├── deploy.py                 # deploy model to SageMaker endpoint
│   ├── predict.py                # call the live endpoint
│   └── ecr_push.sh               # build + push Docker image to ECR
├── docker/
│   └── Dockerfile                # SageMaker training container
├── inference/
│   └── inference.py              # model_fn / input_fn / predict_fn / output_fn
├── train.py                      # main training entrypoint
├── hpo.py                        # Optuna HPO (calls train.py per trial)
├── pytorch_to_onnx.py            # export trained checkpoint to ONNX
├── requirements.txt
└── setup.py
```

---

## Setup

```bash
git clone https://github.com/Hamza-Ali0237/vision-segmentation-platform
cd vision-segmentation-platform

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e .
pip install -r requirements.txt
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
pytest                        # run all tests
pytest tests/test_models.py   # just model shape tests
pytest -m "not slow"          # skip any slow tests
pytest -v --tb=long           # verbose with full tracebacks
```

All tests are CPU-only and use synthetic data — no GPU or real dataset needed.

---

## AWS SageMaker Deployment

### Prerequisites

1. AWS account with ~$100 free credits
2. AWS CLI installed and configured (`aws configure`)
3. Docker installed locally
4. Fill in `training/configs/base.yaml`:
   ```yaml
   aws:
     bucket:        "your-s3-bucket-name"
     role_arn:      "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
     ecr_image_uri: ""   # filled in after step 2 below
   ```

### Step 1 — Create the IAM Role

In the AWS Console → IAM → Roles → Create Role:
- Trusted entity: **SageMaker**
- Attach policies: `AmazonSageMakerFullAccess`, `AmazonS3FullAccess`
- Name it `SageMakerExecutionRole`
- Copy the ARN into `base.yaml`

### Step 2 — Push Docker image to ECR

```bash
chmod +x aws/ecr_push.sh
./aws/ecr_push.sh 123456789012 us-east-1 cxr-segmentation
```

Copy the printed URI into `base.yaml → aws.ecr_image_uri`.

### Step 3 — Upload dataset to S3

```bash
aws s3 sync data/images s3://your-bucket/data/images/
aws s3 sync data/masks  s3://your-bucket/data/masks/
```

### Step 4 — Launch training job

```bash
pip install sagemaker boto3
python aws/train_job.py --arch unet
```

This blocks until the job finishes (up to 4 hours — the hard wall-clock limit protects your credits). Training logs stream to your terminal.

### Step 5 — Deploy endpoint

```bash
python aws/deploy.py --arch unet
```

### Step 6 — Run a prediction

```bash
python aws/predict.py --image path/to/xray.png --arch unet
```

### Step 7 — Delete the endpoint when done

**Important** — endpoints bill by the hour even when idle:

```bash
aws sagemaker delete-endpoint --endpoint-name cxr-seg-unet-endpoint
```

---

## Export to ONNX

```bash
python pytorch_to_onnx.py \
  --checkpoint outputs/models/best_unet.pth \
  --arch unet \
  --output outputs/unet.onnx
```

---

## MLflow Quick Reference

| What | Command |
|---|---|
| View runs locally | `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| Compare two runs | UI → select runs → Compare |
| Download artifact | UI → run → Artifacts tab |
| List experiments | `mlflow experiments list` |

---