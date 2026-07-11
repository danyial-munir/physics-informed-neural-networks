"""
Solving the Fisher-KPP equation for a sweep of (D, r) combinations.

Usage:
    python pinn_cases.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
from config import Config
from data import generate_data
from model import FCNet
from trainer import train
from plot import save_plots_from_file
from utils import get_device, save_model

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "outputs"
OUTPUT_PATH.mkdir(exist_ok=True)


def train_model(cfg: Config, output_path: Path, device) -> None:
    """Train and save one (D, r) case."""
    print(cfg)

    data = generate_data(cfg)
    model = FCNet(cfg)
    history, snapshots, best_state = train(
        model=model,
        data=data,
        cfg=cfg,
        device=device,
        label="Model",
    )

    save_model(best_state, history, snapshots, cfg, output_path)
    print(f"Model saved to {output_path}")

    save_plots_from_file(output_path)


def main() -> None:
    """Main loop."""
    device = get_device()
    print(f"Device : {device}")

    D_values = [0.5, 1.0, 2.0]
    r_values = [0.5, 1.0, 2.0]

    for D in D_values:
        for r in r_values:
            cfg = Config(D=D, r=r)
            label = f"D{D:g}_r{r:g}".replace(".", "p")
            output_path = OUTPUT_PATH / label
            train_model(cfg, output_path, device)


if __name__ == "__main__":
    main()
