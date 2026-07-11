"""
Plotting utilities for the Fisher-KPP PINN.

Contains:
    style_ax                            -- shared axis styling
    plot_solution_heatmap                -- PINN vs FD reference heatmap
    plot_snapshots                       -- PINN vs FD reference at fixed times
    plot_losses                          -- training loss curves
    plot_predicted_parameter_convergence -- D_hat, r_hat vs epoch (inverse mode)
    save_plots_from_file                 -- load a saved run and regenerate all plots
"""

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from config import Config
from analytic import wave_speed, solve_kpp_fd
from model import FCNet, InverseFCNet, predict_from_state
from utils import save_show, load_model

BLUE = "#378ADD"
RED = "#E24B4A"
GREEN = "#1D9E75"
ORANGE = "#EF9F27"
PURPLE = "#7F77DD"
GRAY = "#888780"
LGRAY = "#D3D1C7"
BG = "#FAFAF8"
PANEL = "#F1EFE8"


def style_ax(ax: Axes) -> None:
    """Apply shared axis styling."""
    ax.set_facecolor(PANEL)
    ax.figure.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(LGRAY)
    ax.tick_params(colors=GRAY)


def plot_solution_heatmap(
    model_state: dict,
    cfg: Config,
    u_grid: np.ndarray,
    t_arr: np.ndarray,
    output_path: str = None,
    show: bool = True,
) -> None:
    """Side-by-side heatmap of the FD reference and the PINN prediction."""
    n_t, n_x = u_grid.shape
    x = np.linspace(-cfg.L, cfg.L, n_x)

    tt, xx = np.meshgrid(t_arr, x, indexing="ij")
    u_pred = predict_from_state(model_state, tt.ravel(), xx.ravel(), cfg).reshape(n_t, n_x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    im = None
    for ax, grid, title in zip(axes, [u_grid, u_pred], ["FD reference", "PINN"]):
        style_ax(ax)
        im = ax.pcolormesh(x, t_arr, grid, shading="auto", cmap="viridis")
        ax.set_xlabel("$x$")
        ax.set_title(title)
    axes[0].set_ylabel("$t$")
    fig.colorbar(im, ax=axes, label="$u(x,t)$")

    save_show(output_path, show)


def plot_snapshots(
    model_state: dict,
    cfg: Config,
    u_grid: np.ndarray,
    t_arr: np.ndarray,
    times: tuple = (0.0, 1.5, 3.0, 4.5),
    output_path: str = None,
    show: bool = True,
) -> None:
    """Plot FD reference vs PINN prediction at fixed times, plus the analytic wave-speed marker."""
    n_x = u_grid.shape[1]
    x = np.linspace(-cfg.L, cfg.L, n_x)
    c = wave_speed(cfg)

    fig, ax = plt.subplots(figsize=(9, 6))
    style_ax(ax)
    for t_target in times:
        idx = int(np.argmin(np.abs(t_arr - t_target)))
        t_actual = t_arr[idx]
        u_pred = predict_from_state(model_state, np.full_like(x, t_actual), x, cfg)

        ax.plot(x, u_grid[idx], color=GRAY, linestyle="--", alpha=0.7)
        ax.plot(x, u_pred, color=GREEN, alpha=0.9, label=f"$t={t_actual:.2f}$")
        ax.axvline(c * t_actual, color=RED, linestyle=":", alpha=0.4)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(x,t)$")
    ax.set_title("Fisher-KPP: PINN (solid) vs FD reference (dashed)")
    ax.legend(fontsize=8)

    save_show(output_path, show)


def plot_losses(history: dict, output_path: str = None, show: bool = True) -> None:
    """Plot training loss curves on a log scale."""
    fig, ax = plt.subplots(figsize=(8, 5))
    style_ax(ax)
    for key, color in [
        ("loss_data", BLUE),
        ("loss_phys", ORANGE),
        ("loss_ic", PURPLE),
        ("loss_bc", RED),
        ("loss_val", GREEN),
        ("loss_total", GRAY),
    ]:
        if any(v != 0 for v in history[key]):
            ax.plot(history["epoch"], history[key], label=key, color=color)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)

    save_show(output_path, show)


def plot_predicted_parameter_convergence(
    history: dict, cfg: Config, output_path: str = None, show: bool = True
) -> None:
    """Plot D_hat and r_hat convergence against the ground-truth values."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, true_val, label in zip(
        axes, ["D_hat", "r_hat"], [cfg.D, cfg.r], ["$D$", "$r$"]
    ):
        style_ax(ax)
        ax.plot(history["epoch"], history[key], color=GREEN, label=f"{label} (predicted)")
        ax.axhline(true_val, color=RED, linestyle="--", label=f"{label} (true)")
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)

    save_show(output_path, show)


def _peek_history_keys(folder_path: Path) -> list:
    """Read just the header row of history.csv to detect inverse-mode columns."""
    with open(folder_path / "history.csv", encoding="utf-8") as f:
        return next(csv.reader(f))


def save_plots_from_file(folder_path: str, verbatim: bool = True) -> None:
    """Load a saved run (best_model.pt, history.csv, config.json) and regenerate all plots."""
    folder_path = Path(folder_path)

    with open(folder_path / "config.json", encoding="utf-8") as f:
        cfg = Config(**json.load(f))

    inverse_mode = "D_hat" in _peek_history_keys(folder_path)
    model = InverseFCNet(cfg) if inverse_mode else FCNet(cfg)
    model, history, _snapshots, cfg = load_model(model, folder_path)

    u_grid, t_arr = solve_kpp_fd(cfg)
    best_state = model.state_dict()

    plot_solution_heatmap(
        best_state, cfg, u_grid, t_arr,
        output_path=str(folder_path / "solution_heatmap.png"), show=False,
    )
    plot_snapshots(
        best_state, cfg, u_grid, t_arr,
        output_path=str(folder_path / "snapshots.png"), show=False,
    )
    plot_losses(history, output_path=str(folder_path / "losses.png"), show=False)

    if inverse_mode:
        plot_predicted_parameter_convergence(
            history, cfg,
            output_path=str(folder_path / "parameter_convergence.png"), show=False,
        )

    if verbatim:
        print(f"Plots saved to {folder_path}/")
