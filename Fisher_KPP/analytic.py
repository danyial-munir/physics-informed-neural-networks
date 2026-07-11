"""
Initial condition and finite-difference reference solver for the
Fisher-KPP equation: u_t = D*u_xx + r*u*(1-u).
"""

import numpy as np
from tqdm import tqdm
from config import Config


def u_0(cfg: Config) -> np.ndarray:
    """Smoothed step initial condition: 0.5*(1 - tanh(x/eps))."""
    x = np.linspace(-cfg.L, cfg.L, cfg.n_x)
    return 0.5 * (1.0 - np.tanh(x / cfg.eps))


def wave_speed(cfg: Config) -> float:
    """Analytic KPP traveling-wave speed: c = 2*sqrt(r*D)."""
    return 2.0 * np.sqrt(cfg.r * cfg.D)


def compute_stable_dt(cfg: Config, cfl: float = 0.4) -> float:
    """
    Stable explicit time step for the diffusion term (von Neumann):
    dt <= dx^2 / (2*D). The reaction term r*u*(1-u) is bounded
    (|r*u*(1-u)| <= r/4 for u in [0,1]) so it does not tighten the
    diffusion-dominated stability limit for the parameter ranges used here.
    """
    dx = cfg.delta_x
    return cfl * dx**2 / cfg.D


def fd_step(u: np.ndarray, dt: float, cfg: Config) -> np.ndarray:
    """
    One explicit-Euler FD step:
        u_xx via central differences (interior points)
        reaction term r*u*(1-u) explicit
        Dirichlet BC: u[0] = 1, u[-1] = 0 held fixed every step
    """
    dx = cfg.delta_x
    u_next = u.copy()

    laplacian = np.zeros_like(u)
    laplacian[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2

    reaction = cfg.r * u * (1.0 - u)

    u_next[1:-1] = u[1:-1] + dt * (cfg.D * laplacian[1:-1] + reaction[1:-1])
    u_next[0] = 1.0
    u_next[-1] = 0.0

    return u_next


def solve_kpp_fd(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """
    March the FD solver from u_0 to t=T using a CFL-stable dt,
    recording n_t evenly spaced snapshots (via linear-in-index sampling
    of the internal fine time steps).
    """
    dt = compute_stable_dt(cfg)
    n_steps = max(int(np.ceil(cfg.T / dt)), 1)
    dt = cfg.T / n_steps  # exact fit to T

    u = u_0(cfg)
    snapshot_every = max(n_steps // cfg.n_t, 1)

    u_grid = [u.copy()]
    t_list = [0.0]

    for step in tqdm(range(1, n_steps + 1), desc="Computing FD reference solution"):
        u = fd_step(u, dt, cfg)
        if step % snapshot_every == 0 or step == n_steps:
            u_grid.append(u.copy())
            t_list.append(step * dt)

    return np.array(u_grid), np.array(t_list)


def interpolate_solution(
    u_grid: np.ndarray, t_arr: np.ndarray, x: float, t: float, cfg: Config
) -> float:
    """Bilinear interpolation of the FD solution at physical coordinates (x, t)."""
    n_t, n_x = u_grid.shape

    xi = (x + cfg.L) / cfg.delta_x
    ti = np.searchsorted(t_arr, t, side="right") - 1
    ti = np.clip(ti, 0, n_t - 2)
    xi = np.clip(xi, 0, n_x - 1)

    x0, x1 = int(np.floor(xi)), min(int(np.floor(xi)) + 1, n_x - 1)
    t0, t1 = int(ti), min(int(ti) + 1, n_t - 1)

    dx = xi - x0
    dt = (t - t_arr[t0]) / (t_arr[t1] - t_arr[t0]) if t_arr[t1] != t_arr[t0] else 0.0

    f00, f10 = u_grid[t0, x0], u_grid[t0, x1]
    f01, f11 = u_grid[t1, x0], u_grid[t1, x1]

    return (
        f00 * (1 - dx) * (1 - dt)
        + f10 * dx * (1 - dt)
        + f01 * (1 - dx) * dt
        + f11 * dx * dt
    )
