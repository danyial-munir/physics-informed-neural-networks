import numpy as np
import torch

from config import Config
from model import FCNet, InverseFCNet, predict, predict_from_state


def small_cfg(**overrides) -> Config:
    defaults = dict(n_x=100, T=1.0, n_t=20, n_obs=5, n_col=10, n_bc=5, n_grid_val=5)
    defaults.update(overrides)
    return Config(**defaults)


def test_forward_net_shape():
    cfg = small_cfg()
    model = FCNet(cfg)
    t = torch.rand(8, 1)
    x = torch.rand(8, 1)
    out = model(t, x)
    assert out.shape == (8, 1)
    assert torch.isfinite(out).all()


def test_inverse_net_forward_and_params_positive():
    cfg = small_cfg()
    model = InverseFCNet(cfg)
    t = torch.rand(8, 1)
    x = torch.rand(8, 1)
    out = model(t, x)
    assert out.shape == (8, 1)
    assert model.D_hat.item() > 0
    assert model.r_hat.item() > 0


def test_predict_matches_numpy_shape():
    cfg = small_cfg()
    model = FCNet(cfg)
    t = np.linspace(0.0, 1.0, 20)
    x = np.linspace(-cfg.L, cfg.L, 20)
    y = predict(model, t, x)
    assert y.shape == (20,)
    assert np.isfinite(y).all()


def test_predict_from_state_matches_direct_predict():
    cfg = small_cfg()
    model = FCNet(cfg)
    t = np.linspace(0.0, 1.0, 10)
    x = np.linspace(-cfg.L, cfg.L, 10)

    state_dict = model.state_dict()
    u_direct = predict(model, t, x)
    u_restored = predict_from_state(state_dict, t, x, cfg)

    assert np.allclose(u_direct, u_restored)


def test_predict_from_state_handles_inverse_model_state_dict():
    cfg = small_cfg()
    model = InverseFCNet(cfg)
    t = np.linspace(0.0, 1.0, 10)
    x = np.linspace(-cfg.L, cfg.L, 10)

    state_dict = model.state_dict()
    u = predict_from_state(state_dict, t, x, cfg)
    assert u.shape == (10,)
    assert np.isfinite(u).all()
