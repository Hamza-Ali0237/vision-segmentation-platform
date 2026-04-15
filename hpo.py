"""
Hyperparameter optimisation with Optuna.
Each trial is logged as a nested MLflow run under a parent "hpo" run.

Usage:
    python hpo.py --config training/configs/base.yaml \
                  --arch unet \
                  --images-dir data/images \
                  --masks-dir  data/masks \
                  --n-trials 20
"""

import argparse
import os
import sys
import yaml
import mlflow
import optuna
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train import run_training, _noop


# Optuna objective
def make_objective(config: dict, arch: str, images_dir: str, masks_dir: str, parent_run_id: str):
    def objective(trial: optuna.Trial) -> float:
        ss = config.get("optuna", {}).get("search_space", {})

        override = {
            "lr": trial.suggest_float("lr",
                                ss.get("lr", [1e-5, 1e-3])[0],
                                ss.get("lr", [1e-5, 1e-3])[1],
                                log=True),
            "weight_decay": trial.suggest_float("weight_decay",
                                ss.get("weight_decay", [1e-5, 1e-3])[0],
                                ss.get("weight_decay", [1e-5, 1e-3])[1],
                                log=True),
            "weight_bce": trial.suggest_float("weight_bce",
                                ss.get("weight_bce", [0.5, 2.0])[0],
                                ss.get("weight_bce", [0.5, 2.0])[1]),
            "weight_dice": trial.suggest_float("weight_dice",
                                ss.get("weight_dice", [0.5, 2.0])[0],
                                ss.get("weight_dice", [0.5, 2.0])[1]),
            "batch_size": trial.suggest_categorical("batch_size",
                                ss.get("batch_size", [4, 8, 16])),
            "lr_patience": trial.suggest_categorical("lr_patience",
                                ss.get("lr_patience", [5, 10, 15])),
        }

        # Patch batch_size into config key training.batch_size (loaders read it)
        config["training"]["batch_size"] = override["batch_size"]

        with mlflow.start_run(
            run_name=f"trial_{trial.number}",
            nested=True,
            tags={"trial_number": str(trial.number)},
        ):
            mlflow.log_params({"trial_number": trial.number, **override})
            val_loss = run_training(
                config=config,
                arch=arch,
                images_dir=images_dir,
                masks_dir=masks_dir,
                override=override,
                mlflow_run=mlflow.active_run(),
            )
            mlflow.log_metric("val_loss", val_loss)

        return val_loss

    return objective


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description="Optuna HPO for CXR segmentation")
    parser.add_argument("--config", default="training/configs/base.yaml")
    parser.add_argument("--arch", default="unet", choices=["unet", "segnet"])
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--study-name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    optuna_cfg = config.get("optuna", {})
    n_trials = args.n_trials or optuna_cfg.get("n_trials", 20)
    direction = optuna_cfg.get("direction", "minimize")
    use_pruning = optuna_cfg.get("pruning", True)
    study_name = args.study_name or f"hpo-{args.arch}"

    # MLflow: one parent run wraps all trials
    mlflow_cfg = config.get("mlflow", {})
    experiment_name = mlflow_cfg.get("experiment_name", "cxr-segmentation")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5) \
             if use_pruning else optuna.pruners.NopPruner()

    # Persist the study in a local SQLite DB so you can resume interrupted runs
    storage = f"sqlite:///optuna_{args.arch}.db"
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    with mlflow.start_run(run_name=f"hpo-{args.arch}") as parent_run:
        mlflow.log_params({
            "n_trials": n_trials,
            "direction": direction,
            "arch": args.arch,
            "study_name": study_name,
        })

        objective = make_objective(
            config=config,
            arch=args.arch,
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            parent_run_id=parent_run.info.run_id,
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Log best trial summary back to the parent run
        best = study.best_trial
        mlflow.log_metrics({
            "best_val_loss":   best.value,
            "best_trial_number": best.number,
        })
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})

    print("\n" + "=" * 60)
    print(f"Best trial:  #{best.number}")
    print(f"Best val loss: {best.value:.4f}")
    print("Best hyperparams:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Save best params to YAML for easy reference
    best_params_path = f"outputs/best_params_{args.arch}.yaml"
    Path(best_params_path).parent.mkdir(parents=True, exist_ok=True)
    with open(best_params_path, "w") as f:
        yaml.dump({"best_trial": best.number, "params": best.params}, f)
    print(f"Best params saved to {best_params_path}")


if __name__ == "__main__":
    main()