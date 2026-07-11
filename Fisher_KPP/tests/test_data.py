import numpy as np

from config import Config
from analytic import u_0, solve_kpp_fd
from data import make_observation, make_collocation, make_bc_points, make_validation, generate_data


def small_cfg(**overrides) -> Config:
    defaults = dict(n_x=100, T=1.0, n_t=20, n_obs=10, n_col=20, n_bc=10, n_grid_val=10)
    defaults.update(overrides)
    return Config(**defaults)


def test_make_collocation_within_domain_bounds():
    cfg = small_cfg()
    t_col, x_col = make_collocation(cfg)
    assert t_col.shape == x_col.shape == (cfg.n_col,)
    assert (t_col >= 0).all() and (t_col <= cfg.T).all()
    assert (x_col >= -cfg.L).all() and (x_col <= cfg.L).all()


def test_make_bc_points_within_time_domain():
    cfg = small_cfg()
    t_bc = make_bc_points(cfg)
    assert t_bc.shape == (cfg.n_bc,)
    assert (t_bc >= 0).all() and (t_bc <= cfg.T).all()


def test_make_observation_interpolates_within_grid_range():
    cfg = small_cfg(sigma=0.0)
    u_grid, t_arr = solve_kpp_fd(cfg)
    t_obs, x_obs, u_obs = make_observation(cfg, u_grid, t_arr)
    assert t_obs.shape == x_obs.shape == u_obs.shape == (cfg.n_obs,)
    assert np.isfinite(u_obs).all()


def test_make_validation_grid_shapes():
    cfg = small_cfg()
    u_grid, t_arr = solve_kpp_fd(cfg)
    t_val, x_val, u_val = make_validation(cfg, u_grid, t_arr)
    assert t_val.shape == x_val.shape == u_val.shape == (cfg.n_grid_val**2,)
    assert np.isfinite(u_val).all()


def test_generate_data_shapes_and_ic_matches_u_0():
    cfg = small_cfg()
    data = generate_data(cfg)
    assert data["t_obs"].shape == data["x_obs"].shape == data["u_obs"].shape
    assert data["t_val"].shape == data["x_val"].shape == data["u_val"].shape
    assert data["u_ic"].shape == data["x_ic"].shape
    assert np.allclose(data["u_ic"], u_0(cfg))
    assert np.isfinite(data["u_grid"]).all()
