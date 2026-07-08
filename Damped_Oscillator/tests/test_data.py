import numpy as np
import pytest

from config import Config
from data import analytic, make_train_observation, make_validation, make_collocation


@pytest.mark.parametrize("zeta,c", [(0.5, 2.0), (1.0, 4.0), (2.0, 8.0)])
def test_analytic_satisfies_initial_conditions(zeta, c):
    """y(0) == y0 and y'(0) == dy0 for under/critically/over-damped regimes."""
    cfg = Config(m=1.0, k=4.0, c=c, y0=1.0, dy0=0.5)
    assert np.isclose(cfg.zeta, zeta)

    t = np.array([0.0])
    y0 = analytic(t, cfg)
    assert np.isclose(y0[0], cfg.y0, atol=1e-8)

    dt = 1e-6
    dy_fd = (analytic(np.array([dt]), cfg) - analytic(np.array([0.0]), cfg)) / dt
    assert np.isclose(dy_fd[0], cfg.dy0, atol=1e-3)


def test_analytic_decays_toward_zero_over_time():
    cfg = Config(y0=1.0, dy0=0.0)
    y_late = analytic(np.array([50.0]), cfg)
    assert abs(y_late[0]) < 1e-3


def test_make_train_observation_matches_analytic_within_noise():
    cfg = Config(sigma=0.0, n_obs=25)
    t_obs, y_obs = make_train_observation(cfg)
    assert np.allclose(y_obs, analytic(t_obs, cfg))


def test_make_collocation_domains_respect_time_bounds():
    cfg = Config(t_dom=2.0, t_extrap=3.0, n_col_dom=50)
    t_col_dom, t_col_extrap = make_collocation(cfg)
    assert (t_col_dom >= 0.1).all() and (t_col_dom <= cfg.t_dom).all()
    assert (t_col_extrap >= cfg.t_dom).all() and (t_col_extrap <= cfg.t_extrap).all()


def test_make_validation_is_deterministic_grid():
    cfg = Config(n_val=10)
    t_val_a, y_val_a = make_validation(cfg)
    t_val_b, y_val_b = make_validation(cfg)
    assert np.array_equal(t_val_a, t_val_b)
    assert np.array_equal(y_val_a, y_val_b)
