"""
Plot generation for the Fisher-KPP report.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
import shutil
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from analytic import wave_speed, solve_kpp_fd
from plot import style_ax

USE_TEX = shutil.which("latex") is not None
plt.rcParams.update(
    {"text.usetex": USE_TEX, "font.family": "Helvetica" if USE_TEX else "DejaVu Sans"}
)

REPORT_DIR = Path(__file__).resolve().parent.parent / "outputs/report_images"


def ic_and_wave_speed_figure():
    """Plot the FD solution at several times, with the analytic wave-speed
    front position marked."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = Config(n_x=400, T=6.0, n_t=600)
    x = np.linspace(-cfg.L, cfg.L, cfg.n_x)
    c = wave_speed(cfg)

    u_grid, t_arr = solve_kpp_fd(cfg)

    fig, ax = plt.subplots(figsize=(10, 6))
    style_ax(ax)
    for t_target in (0.0, 2.0, 4.0, 6.0):
        idx = int(np.argmin(np.abs(t_arr - t_target)))
        ax.plot(x, u_grid[idx], label=f"$t={t_arr[idx]:.1f}$")
        ax.axvline(c * t_arr[idx], linestyle=":", alpha=0.4)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    ax.set_xlabel("$x$", fontsize=14)
    ax.set_ylabel(r"Solution $u(x,\ t)$", fontsize=14)
    ax.set_title(r"Fisher-KPP traveling front ($D=1$, $r=1$, $c=2$)", fontsize=16)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "kpp_traveling_front.png")


def main():
    ic_and_wave_speed_figure()


if __name__ == "__main__":
    main()
