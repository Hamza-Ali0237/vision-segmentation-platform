import os
import sys
import argparse
import random
import shutil
import yaml
import torch
import mlflow
import numpy as np
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loaders import get_loaders
from src.scripts.protocols import build_model, training_setup
from src.scripts.metrics import get_metrics
from src.scripts.train_logic import train_one_epoch
from src.scripts.validate import validate_one_epoch
from src.scripts.visualize import save_predictions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


# ---------------------------------------------------------------------------
# Core train function (also called by Optuna trials)
# ---------------------------------------------------------------------------

def run_training(
    config,
    arch,
    images_dir,
    masks_dir,
    override=None, # Optuna injects hyperparams here
    mlflow_run=None, # pass an active run when called from hpo.py
):
    
    override = override or {}
    set_seed(config["project"]["seed"])
    device = get_device()

    # Merge overrides into a working copy of config so we don't mutate original
    cfg = yaml.safe_load(yaml.dump(config))
    for k, v in override.items():
        if k in cfg["training"]:
            cfg["training"][k] = v

    print(f"\n{'='*60}")
    print(f"  Architecture : {arch.upper()}")
    print(f"  Device       : {device}")
    print(f"  Batch size   : {cfg['training']['batch_size']}")
    print(f"  LR           : {cfg['training']['learning_rate']}")
    print(f"{'='*60}\n")

    # ---- Data ----------------------------------------------------------------
    train_loader, val_loader, test_loader = get_loaders(cfg, images_dir, masks_dir)

    # ---- Model / optimiser ---------------------------------------------------
    model = build_model(arch, cfg).to(device)
    criterion, optimizer, scheduler = training_setup(model, cfg, override)
    metrics = get_metrics(cfg["model"]["num_classes"], device)

    # ---- Paths ---------------------------------------------------------------
    model_dir  = cfg["paths"].get("model_dir",  "outputs/models")
    output_dir = cfg["paths"].get("output_dir", "outputs")
    ckpt_path  = os.path.join(model_dir, f"best_{arch}.pth")
    vis_dir    = os.path.join(output_dir, "visualisations", arch)

    # ---- MLflow logging ------------------------------------------------------
    own_run = mlflow_run is None
    if own_run:
        mlflow_cfg = config.get("mlflow", {})
        experiment_name = mlflow_cfg.get("experiment_name", "cxr-segmentation")
        mlflow.set_experiment(experiment_name)
        active_run = mlflow.start_run(run_name=f"{arch}-training")
    else:
        active_run = mlflow_run

    with active_run if own_run else _noop():
        # Log all hyperparams
        mlflow.log_params({
            "arch": arch,
            "epochs": cfg["training"]["epochs"],
            "batch_size": cfg["training"]["batch_size"],
            "lr": cfg["training"]["learning_rate"],
            "weight_decay": cfg["training"]["weight_decay"],
            "weight_bce": cfg["training"].get("weight_bce", 1.0),
            "weight_dice": cfg["training"].get("weight_dice", 1.0),
            "image_size": cfg["data"]["image_size"],
            "augmentation": cfg["augmentation"]["apply"],
            **override,
        })
        mlflow.log_artifact("training/configs/base.yaml", artifact_path="config")

        best_val_loss = float("inf")
        patience_counter = 0
        early_stop_patience = cfg["training"].get("early_stopping_patience", 15)

        for epoch in range(1, cfg["training"]["epochs"] + 1):
            train_loss, train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, metrics, device
            )
            val_loss, val_metrics = validate_one_epoch(
                model, val_loader, criterion, metrics, device
            )
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]

            # --- per-epoch MLflow metrics ---
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_dice": train_metrics.get("dice_score", 0),
                    "train_miou": train_metrics.get("miou", 0),
                    "val_dice": val_metrics.get("dice_score", 0),
                    "val_miou": val_metrics.get("miou", 0),
                    "learning_rate": current_lr,
                },
                step=epoch,
            )

            print(
                f"Epoch [{epoch:3d}/{cfg['training']['epochs']}] "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}  "
                f"val_dice={val_metrics.get('dice_score', 0):.4f}  "
                f"val_miou={val_metrics.get('miou', 0):.4f}  "
                f"lr={current_lr:.2e}"
            )

            # --- checkpoint on improvement ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "arch": arch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
                        "config": cfg,
                    },
                    ckpt_path,
                )
                mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        # --- log best checkpoint as MLflow artifact ---
        if os.path.exists(ckpt_path):
            mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

        # --- visualise predictions on val set ---
        vis_paths = save_predictions(model, val_loader, device, vis_dir, num_samples=8)
        for vp in vis_paths:
            mlflow.log_artifact(vp, artifact_path="visualisations")

        # --- final test evaluation ---
        test_loss, test_metrics = validate_one_epoch(
            model, test_loader, criterion, metrics, device
        )
        mlflow.log_metrics(
            {
                "test_loss": test_loss,
                "test_dice": test_metrics.get("dice_score", 0),
                "test_miou": test_metrics.get("miou", 0),
            }
        )
        print(
            f"\nTest results — loss={test_loss:.4f}  "
            f"dice={test_metrics.get('dice_score', 0):.4f}  "
            f"miou={test_metrics.get('miou', 0):.4f}"
        )

    return best_val_loss


# ---------------------------------------------------------------------------
# Context manager shim so "with _noop():" works when caller owns the run
# ---------------------------------------------------------------------------
from contextlib import contextmanager

@contextmanager
def _noop():
    yield


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet / SegNet on chest X-rays")
    parser.add_argument("--config", default="training/configs/base.yaml")
    parser.add_argument("--arch", default="unet", choices=["unet", "segnet", "all"])
    parser.add_argument("--images-dir", default=None,
                        help="Override config paths.data_dir/images")
    parser.add_argument("--masks-dir", default=None,
                        help="Override config paths.data_dir/masks")
    # SageMaker passes hyperparameters as CLI args too
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # SageMaker-style CLI overrides
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr

    # Data paths: CLI > env var (SageMaker) > config
    data_root = config["paths"]["data_dir"]
    images_dir = args.images_dir or os.environ.get("SM_CHANNEL_TRAINING",
                    os.path.join(data_root, "images"))
    masks_dir = args.masks_dir or os.path.join(
                    os.path.dirname(images_dir), "masks")

    # MLflow setup — use S3 artifact store, local SQLite tracking
    mlflow_cfg = config.get("mlflow", {})
    artifact_location = mlflow_cfg.get("artifact_location", "")
    if artifact_location:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    archs = config["model"]["architectures"] if args.arch == "all" else [args.arch]

    for arch in archs:
        run_training(config, arch, images_dir, masks_dir)

    # If running inside SageMaker, copy mlflow.db to output dir so it's preserved
    if os.path.exists("/opt/ml/output") and os.path.exists("mlflow.db"):
        shutil.copy("mlflow.db", "/opt/ml/output/mlflow.db")
        print("MLflow DB copied to /opt/ml/output/mlflow.db")


if __name__ == "__main__":
    main()