import numpy as np
import pytest

from config import Config
from analytic import u_0, predict_shock_time
from data import make_observation, make_collocation, make_bc_points


@pytest.mark.parametrize(
    "ic", ["Gauss", "Step_up", "Step_down", "N_wave", "N_wave_chop", "Slope"]
)
def test_u_0_shapes_and_bounds_for_every_ic_type(ic):
    cfg = Config(ic=ic, n_x=200)
    u_init = u_0(cfg)
    assert u_init.shape == (cfg.n_x,)
    assert np.isfinite(u_init).all()
    assert np.max(np.abs(u_init)) <= cfg.height + 1e-8


def test_unknown_ic_type_raises():
    with pytest.raises(ValueError):
        Config(ic="not_a_real_ic")


def test_predict_shock_time_finite_for_n_wave():
    cfg = Config(ic="N_wave", n_x=200)
    t_shock = predict_shock_time(cfg)
    assert t_shock > 0
    assert np.isfinite(t_shock)


def test_predict_shock_time_positive_for_step_down():
    """Step_down has a negative slope everywhere -> finite, positive shock time."""
    cfg = Config(ic="Step_down", n_x=200)
    t_shock = predict_shock_time(cfg)
    assert t_shock > 0
    assert np.isfinite(t_shock)


def test_make_collocation_domains_respect_bounds():
    cfg = Config(t_dom=1.0, t_extrap=1.5, n_col_dom=50, L=10.0)
    t_dom, x_dom, t_extrap, x_extrap = make_collocation(cfg)
    assert (t_dom >= 0.1).all() and (t_dom <= cfg.t_dom).all()
    assert (x_dom >= 0).all() and (x_dom <= cfg.L).all()
    assert (t_extrap >= cfg.t_dom).all() and (t_extrap <= cfg.t_extrap).all()


def test_make_bc_points_within_time_domain():
    cfg = Config(t_dom=2.0, n_bc=30)
    t_bc = make_bc_points(cfg)
    assert t_bc.shape == (cfg.n_bc,)
    assert (t_bc >= 0).all() and (t_bc <= cfg.t_dom).all()


def test_make_observation_interpolates_within_grid_range():
    from analytic import cole_hopf_grid

    cfg = Config(ic="Gauss", n_x=100, t_dom=0.5, t_extrap=0.6, n_obs=5, sigma=0.0)
    u_grid, t_arr = cole_hopf_grid(cfg, pad=200)

    t_obs, x_obs, u_obs = make_observation(cfg, u_grid, t_arr)
    assert t_obs.shape == x_obs.shape == u_obs.shape == (cfg.n_obs,)
    assert np.isfinite(u_obs).all()
