"""
Data generation for the Fisher-KPP PINN: observation, collocation,
boundary, initial-condition, and validation points.
"""

import numpy as np
from config import Config
from analytic import u_0, interpolate_solution, solve_kpp_fd


def make_observation(
    cfg: Config, u_grid: np.ndarray, t_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Noisy observation points sampled uniformly over the domain."""
    t_obs = np.random.uniform(0.0, cfg.T, cfg.n_obs)
    x_obs = np.random.uniform(-cfg.L, cfg.L, cfg.n_obs)

    u_obs = np.array(
        [
            interpolate_solution(u_grid, t_arr, x, t, cfg) + np.random.normal(0.0, cfg.sigma)
            for x, t in zip(x_obs, t_obs)
        ]
    )
    return t_obs, x_obs, u_obs


def make_collocation(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """
    PDE residual collocation points sampled uniformly over the domain.

    When cfg.train_extrap is False, points are restricted to [0, t_train] --
    used by the blind-extrapolation experiment to test whether the PINN
    generalises the traveling front beyond its training window.
    """
    t_max = cfg.T if cfg.train_extrap else cfg.t_train
    t_col = np.random.uniform(0.0, t_max, cfg.n_col)
    x_col = np.random.uniform(-cfg.L, cfg.L, cfg.n_col)
    return t_col, x_col


def make_bc_points(cfg: Config) -> np.ndarray:
    """Boundary-condition time points."""
    return np.random.uniform(0.0, cfg.T, cfg.n_bc)


def make_validation(
    cfg: Config, u_grid: np.ndarray, t_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validation grid: interpolate the FD reference at a regular (t, x) mesh."""
    t_1d = np.linspace(0.0, cfg.T, cfg.n_grid_val)
    x_1d = np.linspace(-cfg.L, cfg.L, cfg.n_grid_val)

    tt, xx = np.meshgrid(t_1d, x_1d)
    t_val = tt.ravel()
    x_val = xx.ravel()

    u_val = np.array(
        [interpolate_solution(u_grid, t_arr, x, t, cfg) for x, t in zip(x_val, t_val)]
    )
    return t_val, x_val, u_val


def generate_data(cfg: Config) -> dict:
    """Generate all data and return as a dict."""
    print("Generating data...")
    u_grid, t_arr = solve_kpp_fd(cfg)

    t_obs, x_obs, u_obs = make_observation(cfg, u_grid, t_arr)
    t_col, x_col = make_collocation(cfg)
    t_val, x_val, u_val = make_validation(cfg, u_grid, t_arr)
    t_bc = make_bc_points(cfg)

    x_ic = np.linspace(-cfg.L, cfg.L, cfg.n_x)

    return {
        "u_grid": u_grid,
        "t_arr": t_arr,
        "t_obs": t_obs,
        "x_obs": x_obs,
        "u_obs": u_obs,
        "t_val": t_val,
        "x_val": x_val,
        "u_val": u_val,
        "t_col": t_col,
        "x_col": x_col,
        "t_bc": t_bc,
        "x_ic": x_ic,
        "t_ic": np.zeros_like(x_ic),
        "u_ic": u_0(cfg),
    }
