"""
Plot generation for the report.

Author          : Danyial Munir
Email           : danyial.munir@studio.unibo.it
Last modified   : 24/05/2026
"""

import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from data import analytic
from plot import style_ax
from utils import convert_to_mck

# -- Font configuration ----------------------------------------
USE_TEX = shutil.which("latex") is not None
plt.rcParams.update(
    {
        "text.usetex": USE_TEX,
        "font.family": "Helvetica" if USE_TEX else "DejaVu Sans",
    }
)

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


def analytic_solutions_zeta_underdamped():
    zetas = np.arange(0.2, 0.8, 0.2)
    omega = 2

    t = np.linspace(0, 10, 200)
    fig, ax = plt.subplots(figsize=(10, 6))
    style_ax(ax)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(0, 10)

    for zeta in zetas:
        m, c, k = convert_to_mck(zeta, omega)
        cfg = Config(m=m, c=c, k=k)
        ax.plot(t, analytic(t, cfg), label=rf"$\zeta={zeta:.2f}$")

    ax.legend()
    ax.set_xlabel("Time $t$", fontsize=14)
    ax.set_ylabel("Solution $y(t)$", fontsize=14)
    ax.set_title(r"Different values for $\zeta$ for the underdamped case", fontsize=16)
    plt.tight_layout()
    plt.savefig("Report/Images/analytic_solutions_zeta_underdamped.png")


def analytic_solutions_zeta_overdamped():
    zetas = np.arange(1, 2, 0.2)
    omega = 2

    t = np.linspace(0, 10, 200)
    fig, ax = plt.subplots(figsize=(10, 6))
    style_ax(ax)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(0, 10)

    for zeta in zetas:
        m, c, k = convert_to_mck(zeta, omega)
        cfg = Config(m=m, c=c, k=k)
        ax.plot(t, analytic(t, cfg), label=rf"$\zeta={zeta:.2f}$")

    ax.legend()
    ax.set_xlabel("Time $t$", fontsize=14)
    ax.set_ylabel("Solution $y(t)$", fontsize=14)
    ax.set_title(r"Different values for $\zeta$ for the overdamped case", fontsize=16)
    plt.tight_layout()
    plt.savefig("Report/Images/analytic_solutions_zeta_overdamped.png")


def analytic_solutions_omega_underdamped():
    omegas = np.arange(1, 4, 0.5)
    zeta = 0.125

    t = np.linspace(0, 10, 200)
    fig, ax = plt.subplots(figsize=(10, 6))
    style_ax(ax)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(0, 10)

    for omega_0 in omegas:
        m, c, k = convert_to_mck(zeta, omega_0)
        cfg = Config(m=m, c=c, k=k)
        ax.plot(t, analytic(t, cfg), label=rf"$\omega_0={omega_0:.2f}$")

    ax.legend()
    ax.set_xlabel("Time $t$", fontsize=14)
    ax.set_ylabel("Solution $y(t)$", fontsize=14)
    ax.set_title(
        r"Different values for $\omega_0$ for the underdamped case", fontsize=16
    )  # noqa: E501
    plt.tight_layout()
    plt.savefig("Report/Images/analytic_solutions_omega_underdamped.png")


def analytic_solutions_omega_overdamped():
    omegas = np.arange(1, 4, 0.5)
    zeta = 1.5

    t = np.linspace(0, 10, 200)
    fig, ax = plt.subplots(figsize=(10, 6))
    style_ax(ax)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(0, 10)

    for omega_0 in omegas:
        m, c, k = convert_to_mck(zeta, omega_0)
        cfg = Config(m=m, c=c, k=k)
        ax.plot(t, analytic(t, cfg), label=rf"$\omega_0={omega_0:.2f}$")

    ax.legend()
    ax.set_xlabel("Time $t$", fontsize=14)
    ax.set_ylabel("Solution $y(t)$", fontsize=14)
    ax.set_title(
        r"Different values for $\omega_0$ for the overdamped case", fontsize=16
    )  # noqa: E501
    plt.tight_layout()
    plt.savefig("Report/Images/analytic_solutions_omega_overdamped.png")


def main():
    analytic_solutions_zeta_underdamped()
    analytic_solutions_zeta_overdamped()
    analytic_solutions_omega_underdamped()
    analytic_solutions_omega_overdamped()


if __name__ == "__main__":
    main()
