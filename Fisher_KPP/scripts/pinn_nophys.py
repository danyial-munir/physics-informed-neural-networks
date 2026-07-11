"""
Solving the Fisher-KPP equation without any physics loss terms
(traditional dense network fit to noisy data only).

Usage:
    python pinn_nophys.py
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


def main() -> None:
    """Main loop."""
    device = get_device()
    print(f"Device : {device}")

    cfg = Config(
        D=1.0,
        r=1.0,
        use_data=True,
        use_physics=False,
        use_ic=False,
        use_bc=False,
    )
    print(cfg)

    output_path = OUTPUT_PATH / "D1_r1_ML"

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


if __name__ == "__main__":
    main()
