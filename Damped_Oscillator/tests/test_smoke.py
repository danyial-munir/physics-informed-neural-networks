import numpy as np
import torch

from config import Config
from model import FCNet, InverseFCNet, predict
from data import make_train_observation, make_validation, make_collocation
from losses import loss_data, loss_physics, loss_ic


def test_config_derived_quantities_finite():
    cfg = Config()
    assert np.isfinite(cfg.omega_0)
    assert np.isfinite(cfg.zeta)
    assert np.isfinite(cfg.omega_d)


def test_forward_net_shape():
    cfg = Config()
    model = FCNet(cfg)
    t = torch.rand(8, 1)
    out = model(t)
    assert out.shape == (8, 1)
    assert torch.isfinite(out).all()


def test_inverse_net_forward_and_params():
    cfg = Config()
    model = InverseFCNet(cfg)
    t = torch.rand(8, 1)
    out = model(t)
    assert out.shape == (8, 1)
    assert model.zeta_hat.requires_grad
    assert model.omega_0_hat.requires_grad


def test_predict_matches_numpy_shape():
    cfg = Config()
    model = FCNet(cfg)
    t = np.linspace(0.1, cfg.t_dom, 20)
    y = predict(model, t)
    assert y.shape == (20,)
    assert np.isfinite(y).all()


def test_data_generation_shapes():
    cfg = Config()
    t_obs, y_obs = make_train_observation(cfg)
    assert t_obs.shape == y_obs.shape == (cfg.n_obs,)

    t_val, y_val = make_validation(cfg)
    assert t_val.shape == y_val.shape

    t_col_dom, t_col_extrap = make_collocation(cfg)
    assert t_col_dom.size > 0
    assert t_col_extrap.size > 0


def test_losses_are_finite_and_differentiable():
    cfg = Config()
    model = FCNet(cfg)

    y_pred = torch.rand(5, 1)
    y_obs = torch.rand(5, 1)
    l_data = loss_data(y_pred, y_obs)
    assert torch.isfinite(l_data)

    t_col = torch.rand(5, 1, requires_grad=True)
    l_phys = loss_physics(cfg, model, t_col)
    assert torch.isfinite(l_phys)
    l_phys.backward()

    t_ic = torch.zeros(1, 1, requires_grad=True)
    l_ic = loss_ic(cfg, model, t_ic)
    assert torch.isfinite(l_ic)
