"""
Demo script for training a PINN to solve the forward Fisher-KPP problem.

Usage:
    python pinn_demo.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
import argparse
from config import Config
from data import generate_data
from model import FCNet
from plot import save_plots_from_file
from trainer import train
from utils import get_device, save_model

SCRIPT_DIR = Path(__file__).resolve().parent.parent


def parse_args():
    """Parsing arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=float, default=None, help="Diffusion coefficient")
    parser.add_argument("--r", type=float, default=None, help="Growth rate")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--output", type=Path, default=None, help="Override output directory")
    return parser.parse_args()


def resolve_output_path(args, D, r):
    """Pick an output folder based on D, r unless overridden."""
    if args.output is not None:
        return args.output
    label = f"D{D:g}_r{r:g}".replace(".", "p")
    return SCRIPT_DIR / f"outputs/pinn_{label}"


def main() -> None:
    """Main loop."""
    print("=" * 50)
    print("  PINN Demo — Fisher-KPP Equation")
    print("  Training a physics-informed neural network")
    print("  to solve the forward problem.")
    print("=" * 50)
    device = get_device()
    print(f"Device : {device}")

    args = parse_args()
    D = args.D if args.D is not None else 1.0
    r = args.r if args.r is not None else 1.0
    output_path = resolve_output_path(args, D, r)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"D      : {D:.4f}")
    print(f"r      : {r:.4f}")

    cfg = Config(D=D, r=r)

    data = generate_data(cfg)
    model = FCNet(cfg)
    history, snapshots, best_state = train(
        model=model, data=data, cfg=cfg, device=device, label="Model",
    )

    save_model(best_state, history, snapshots, cfg, output_path)
    print(f"Model saved to {output_path}")

    save_plots_from_file(output_path)


if __name__ == "__main__":
    main()
