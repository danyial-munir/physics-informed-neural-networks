"""
Optuna TPE hyperparameter search for the Fisher-KPP equation.

Usage:
    python optuna_tuning.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
import random
import csv
import json
import numpy as np
import torch
import optuna

from config import Config
from model import FCNet
from trainer import train
from data import generate_data
from utils import get_device

# -- settings --------------------------------------------------------------
SEED = 42
N_TRIALS = 500

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "outputs/optuna_tuning"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_PATH / "optuna_results.csv"
CSV_FIELDS = [
    "trial",
    "state",
    "rmse",
    "hidden",
    "n_layers",
    "lambda_phys",
    "lambda_ic",
    "lambda_bc",
    "lr",
    "scheduler_gamma",
    "scheduler_step",
]


def set_seed(seed: int) -> None:
    """Set seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


_base_cfg = Config()
_base_data = generate_data(_base_cfg)


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: train and return best validation loss."""
    set_seed(SEED)

    cfg = Config(
        D=_base_cfg.D,
        r=_base_cfg.r,
        hidden=trial.suggest_int("hidden", 16, 128, step=16),
        n_layers=trial.suggest_int("n_layers", 2, 10),
        lambda_phys=trial.suggest_float("lambda_phys", 1e-1, 1e2, log=True),
        lambda_ic=trial.suggest_float("lambda_ic", 1e1, 1e3, log=True),
        lambda_bc=trial.suggest_float("lambda_bc", 1e1, 1e3, log=True),
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        scheduler_gamma=trial.suggest_float("scheduler_gamma", 0.3, 0.9),
        scheduler_step=trial.suggest_int("scheduler_step", 1000, 5000, step=1000),
        use_data=True,
    )

    device = get_device()
    model = FCNet(cfg)

    history, _, _ = train(
        model=model,
        data=_base_data,
        cfg=cfg,
        device=device,
        verbatim=False,
        optuna_trial=trial,
    )

    best_val = min(history["loss_val"])

    row = {
        "trial": trial.number,
        "state": "complete",
        "rmse": best_val,
        "hidden": cfg.hidden,
        "n_layers": cfg.n_layers,
        "lambda_phys": cfg.lambda_phys,
        "lambda_ic": cfg.lambda_ic,
        "lambda_bc": cfg.lambda_bc,
        "lr": cfg.lr,
        "scheduler_gamma": cfg.scheduler_gamma,
        "scheduler_step": cfg.scheduler_step,
    }

    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return best_val


def print_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Print trial progress."""
    print(
        f"  Trial {trial.number:>4} | "
        f"State: {trial.state.name:<10} | "
        f"RMSE: {f'{trial.value:.6f}' if trial.value is not None else 'pruned':>12} | "
        f"Best:  {study.best_value:.6f}"
    )


if __name__ == "__main__":
    my_study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
        sampler=optuna.samplers.TPESampler(seed=SEED),
        storage=f"sqlite:///{OUTPUT_PATH}/optuna.db",
        study_name="Fisher_KPP",
        load_if_exists=True,
    )

    my_study.optimize(
        objective,
        n_trials=N_TRIALS,
        callbacks=[print_callback],
    )

    print(f"\nBest value : {my_study.best_value:.6f}")
    print("Best params:")
    for k, v in my_study.best_params.items():
        print(f"  {k:<25} {v}")

    with open(OUTPUT_PATH / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(
            {"best_val": my_study.best_value, "best_params": my_study.best_params},
            f,
            indent=4,
        )

    df = my_study.trials_dataframe()
    df.to_csv(OUTPUT_PATH / "all_trials.csv", index=False)
    print(f"Saved to {OUTPUT_PATH}")
