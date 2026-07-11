import numpy as np

from config import Config
from analytic import (
    u_0,
    compute_stable_dt,
    fd_step,
    solve_kpp_fd,
    interpolate_solution,
    wave_speed,
)


def test_u_0_shape_and_bounds():
    cfg = Config(n_x=200)
    u = u_0(cfg)
    assert u.shape == (cfg.n_x,)
    assert np.isfinite(u).all()
    assert u.min() >= -1e-6
    assert u.max() <= 1 + 1e-6


def test_u_0_is_decreasing_step_from_1_to_0():
    cfg = Config(n_x=200, eps=0.5)
    u = u_0(cfg)
    assert u[0] > 0.9  # left edge near x=-L, u ~ 1
    assert u[-1] < 0.1  # right edge near x=+L, u ~ 0


def test_compute_stable_dt_positive_and_bounded_by_diffusion():
    cfg = Config(n_x=200, D=1.0)
    dt = compute_stable_dt(cfg)
    assert dt > 0
    assert dt <= 0.5 * cfg.delta_x**2 / cfg.D


def test_fd_step_preserves_shape_and_stays_finite():
    cfg = Config(n_x=200)
    u = u_0(cfg)
    dt = compute_stable_dt(cfg)
    u_next = fd_step(u, dt, cfg)
    assert u_next.shape == u.shape
    assert np.isfinite(u_next).all()


def test_fd_step_respects_dirichlet_boundary_values():
    cfg = Config(n_x=200)
    u = u_0(cfg)
    dt = compute_stable_dt(cfg)
    u_next = fd_step(u, dt, cfg)
    assert abs(u_next[0] - 1.0) < 1e-6
    assert abs(u_next[-1] - 0.0) < 1e-6


def test_solve_kpp_fd_returns_grid_and_stays_bounded():
    cfg = Config(n_x=200, T=2.0, n_t=50)
    u_grid, t_arr = solve_kpp_fd(cfg)
    assert u_grid.shape[1] == cfg.n_x
    assert u_grid.shape[0] == t_arr.shape[0]
    assert np.isfinite(u_grid).all()
    assert u_grid.min() >= -0.05
    assert u_grid.max() <= 1.05


def test_interpolate_solution_matches_grid_at_exact_node():
    cfg = Config(n_x=100, T=1.0, n_t=20)
    u_grid, t_arr = solve_kpp_fd(cfg)
    x0 = -cfg.L
    val = interpolate_solution(u_grid, t_arr, x0, float(t_arr[0]), cfg)
    assert abs(val - u_grid[0, 0]) < 1e-6


def test_wave_speed_formula():
    cfg = Config(D=1.0, r=1.0)
    assert abs(wave_speed(cfg) - 2.0) < 1e-9

    cfg2 = Config(D=0.25, r=1.0)
    assert abs(wave_speed(cfg2) - 1.0) < 1e-9
