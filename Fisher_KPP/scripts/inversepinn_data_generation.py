"""
Script to continuously generate true-predicted (D, r) pairs for the
Fisher-KPP inverse problem and save them to a CSV file.

Runs until KeyboardInterrupt. Each iteration randomises D, r, trains an
InverseFCNet, and appends the result to a CSV.

Usage:
    python inversepinn_data_generation.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
import csv
import numpy as np
from plot import plot_predicted_parameter_convergence, plot_solution_heatmap
from config import Config
from data import generate_data
from model import InverseFCNet
from trainer import train
from utils import get_device

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "outputs/parameter_estimation"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_PATH / "D_r_pairs.csv"
CSV_FIELDS = [
    "run", "D_true", "r_true", "D_pred", "r_pred",
    "D_abs_error", "r_abs_error", "best_val_loss",
]

D_RANGE = (0.5, 2.0)
R_RANGE = (0.5, 2.0)


def run_single(run_idx: int, device) -> dict:
    """Randomise D, r, train an InverseFCNet, return result row."""
    D = float(np.random.uniform(*D_RANGE))
    r = float(np.random.uniform(*R_RANGE))

    print(f"-----Randomised D: {D:.4f}, r: {r:.4f}-----")
    cfg = Config(D=D, r=r, use_data=True)

    data = generate_data(cfg)
    model = InverseFCNet(cfg)

    history, snapshots, _ = train(
        model=model, data=data, cfg=cfg, device=device, label=f"run {run_idx}", verbatim=False,
    )

    D_pred = float(model.D_hat.item())
    r_pred = float(model.r_hat.item())
    best_val = min(history["loss_val"])

    print(
        f"  [run {run_idx}] D_true={D:.4f} D_pred={D_pred:.4f}  "
        f"r_true={r:.4f} r_pred={r_pred:.4f}  val={best_val:.5f}"
    )

    model.to("cpu")
    plot_predicted_parameter_convergence(
        history=history, cfg=cfg,
        output_path=f"{OUTPUT_PATH}/convergence_D{D:.4f}_r{r:.4f}.png", show=False,
    )
    plot_solution_heatmap(
        model_state=model.state_dict(), cfg=cfg,
        u_grid=data["u_grid"], t_arr=data["t_arr"],
        output_path=f"{OUTPUT_PATH}/solution_D{D:.4f}_r{r:.4f}.png", show=False,
    )

    return {
        "run": run_idx, "D_true": D, "r_true": r,
        "D_pred": D_pred, "r_pred": r_pred,
        "D_abs_error": abs(D_pred - D), "r_abs_error": abs(r_pred - r),
        "best_val_loss": best_val,
    }


def main():
    """Main loop."""
    device = get_device()
    print(f"Device      : {device}")
    print(f"Output CSV  : {CSV_PATH}")
    print("Press Ctrl+C to stop.\n")

    write_header = not CSV_PATH.exists()
    run_idx = 0

    try:
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()

            while True:
                run_idx += 1
                print(f"--- Run {run_idx} ---")
                try:
                    row = run_single(run_idx, device)
                except (RuntimeError, ValueError, OSError) as e:
                    print(f"  [run {run_idx}] failed: {e} — skipping.")
                    continue

                writer.writerow(row)
                f.flush()

    except KeyboardInterrupt:
        print(f"\nStopped after {run_idx} runs. Results saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
