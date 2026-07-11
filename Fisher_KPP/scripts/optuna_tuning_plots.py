"""
Load the best hyperparameter-tuned model, retrain it, and save plots.

Usage:
    python optuna_tuning_plots.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
from data import generate_data
from model import FCNet
from trainer import train
from utils import get_device, save_model, load_best_cfg
from plot import save_plots_from_file

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "outputs/optuna_tuning"


def main() -> None:
    """Main loop."""
    device = get_device()
    print(f"Device : {device}\n")

    best_params_path = OUTPUT_PATH / "best_params.json"
    if not best_params_path.exists():
        print(f"No best_params.json found at {best_params_path} — run optuna_tuning.py first.")
        return

    cfg = load_best_cfg(best_params_path)
    print(f"Best cfg : {cfg}")

    data = generate_data(cfg)
    model = FCNet(cfg)
    history, snapshots, best_state = train(
        model=model,
        data=data,
        cfg=cfg,
        device=device,
        label="Optuna best",
    )

    save_model(best_state, history, snapshots, cfg, OUTPUT_PATH)
    print(f"Model saved to {OUTPUT_PATH}")

    save_plots_from_file(OUTPUT_PATH)
    print(f"Plots saved to {OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()
