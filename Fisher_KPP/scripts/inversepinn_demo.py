"""
Demo script for training a PINN to solve the inverse Fisher-KPP problem
(recover D, r from noisy synthetic observations).

Usage:
    python inversepinn_demo.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
import argparse
from config import Config
from data import generate_data
from model import InverseFCNet
from plot import save_plots_from_file
from trainer import train
from utils import get_device, save_model

SCRIPT_DIR = Path(__file__).resolve().parent.parent


def parse_args():
    """Parsing arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=float, default=1.0, help="True diffusion coefficient")
    parser.add_argument("--r", type=float, default=1.0, help="True growth rate")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--output", type=Path, default=None, help="Override output directory")
    return parser.parse_args()


def resolve_output_path(args):
    """Pick an output folder based on D, r unless overridden."""
    if args.output is not None:
        return args.output
    label = f"D{args.D:g}_r{args.r:g}".replace(".", "p")
    return SCRIPT_DIR / f"outputs/inversepinn_{label}"


def main() -> None:
    """Main loop."""
    print("=" * 50)
    print("  Inverse PINN Demo — Fisher-KPP Equation")
    print("  Training a physics-informed neural network")
    print("  to solve the inverse problem.")
    print("=" * 50)
    device = get_device()
    print(f"Device : {device}")

    args = parse_args()
    output_path = resolve_output_path(args)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"D      : {args.D:.4f}")
    print(f"r      : {args.r:.4f}")

    cfg = Config(D=args.D, r=args.r, use_data=True)

    data = generate_data(cfg)
    model = InverseFCNet(cfg)
    history, snapshots, best_state = train(
        model=model, data=data, cfg=cfg, device=device, label="Model",
    )

    save_model(best_state, history, snapshots, cfg, output_path)
    print(f"Model saved to {output_path}")

    save_plots_from_file(output_path)


if __name__ == "__main__":
    main()
