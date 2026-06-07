"""
Plot generation for the report.

Author          : Danyial Munir
Email           : danyial.munir@studio.unibo.it
Last modified   : 24/05/2026
"""

import numpy as np
import matplotlib.pyplot as plt
from config import Config
from analytic import u_0, predict_shock_time, cole_hopf_grid
from plot import style_ax, plot_method_of_characteristics

# -- LaTeX font ------------------------------------------------
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

# -- Color palette ------------------------------------------------
BLUE = "#378ADD"  # noisy training observations
RED = "#E24B4A"  # noisy test/validation observations
GREEN = "#1D9E75"  # PINN
ORANGE = "#EF9F27"  # collocation points
PURPLE = "#7F77DD"  # initial condition marker
GRAY = "#888780"  # true solution / neutral
LGRAY = "#D3D1C7"  # spine colour
BG = "#FAFAF8"  # figure background
PANEL = "#F1EFE8"  # axes background


def gauss_characteristics():
    output_path = "Report/Images/gauss_characteristics.png"
    cfg = Config(ic="Gauss")
    cfg.height = 1.2
    u = u_0(cfg)
    x = np.linspace(0, cfg.L, cfg.n_x)
    cfg.t_shock = predict_shock_time(cfg)
    plot_method_of_characteristics(cfg, output_path=output_path, samples=25, show=False)

    output_path = "Report/Images/gauss_ic.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    style_ax(ax)
    ax.plot(x, u)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    ax.set_xlabel("$x$", fontsize=14)
    ax.set_ylabel(r"Solution $u(0,\ t)$", fontsize=14)
    ax.set_title(r"Gauss initial condition", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)

    output_path = "Report/Images/gauss_t.png"
    cfg.nu = 0.01
    sol, t_arr = cole_hopf_grid(cfg, pad=600)
    fig, ax = plt.subplots(figsize=(10, 6))
    style_ax(ax)
    t_target = 1.0
    idx = np.argmin(np.abs(t_arr - t_target))
    ax.plot(x, sol[idx])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    ax.set_xlabel("$x$", fontsize=14)
    ax.set_ylabel(r"Solution $u(0,\ t)$", fontsize=14)
    ax.set_title(r"Gauss at $t=1$", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)


def main():
    gauss_characteristics()


if __name__ == "__main__":
    main()
