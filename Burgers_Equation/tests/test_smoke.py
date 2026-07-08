import numpy as np
import torch

from config import Config
from model import FCNet, InverseFCNet, predict, predict_from_state
from data import generate_data
from losses import loss_data, loss_physics, loss_physics_inverse, loss_ic, loss_bc


def small_cfg(**overrides) -> Config:
    defaults = dict(n_x=100, t_dom=1.0, t_extrap=1.2, n_obs=5, n_col_dom=10, n_bc=5, n_grid_val=5)
    defaults.update(overrides)
    return Config(**defaults)


def test_config_ic_defaults_populated():
    cfg = Config(ic="Gauss")
    assert cfg.height is not None
    assert cfg.bc_left is not None
    assert cfg.bc_right is not None


def test_forward_net_shape():
    cfg = small_cfg()
    model = FCNet(cfg)
    t = torch.rand(8, 1)
    x = torch.rand(8, 1)
    out = model(t, x)
    assert out.shape == (8, 1)
    assert torch.isfinite(out).all()


def test_inverse_net_forward_and_nu_hat_positive():
    cfg = small_cfg()
    model = InverseFCNet(cfg)
    t = torch.rand(8, 1)
    x = torch.rand(8, 1)
    out = model(t, x)
    assert out.shape == (8, 1)
    assert model.nu_hat.item() > 0


def test_predict_matches_numpy_shape():
    cfg = small_cfg()
    model = FCNet(cfg)
    t = np.linspace(0.1, cfg.t_dom, 20)
    x = np.linspace(0, cfg.L, 20)
    y = predict(model, t, x)
    assert y.shape == (20,)
    assert np.isfinite(y).all()


def test_generate_data_shapes():
    cfg = small_cfg()
    data = generate_data(cfg)
    assert data["t_obs"].shape == data["x_obs"].shape == data["u_obs"].shape
    assert data["t_val"].shape == data["x_val"].shape == data["u_val"].shape
    assert data["u_ic"].shape == data["x_ic"].shape
    assert np.isfinite(data["u_grid"]).all()


def test_losses_are_finite_and_differentiable():
    cfg = small_cfg()
    model = FCNet(cfg)

    u_pred = torch.rand(5, 1)
    u_obs = torch.rand(5, 1)
    l_data = loss_data(u_pred, u_obs)
    assert torch.isfinite(l_data)

    t_col = torch.rand(5, 1, requires_grad=True)
    x_col = torch.rand(5, 1, requires_grad=True)
    l_phys = loss_physics(cfg, model, t_col, x_col)
    assert torch.isfinite(l_phys)
    l_phys.backward()

    t_ic = torch.zeros(5, 1)
    x_ic = torch.rand(5, 1)
    u_ic_true = torch.rand(5, 1)
    l_ic = loss_ic(model, t_ic, x_ic, u_ic_true)
    assert torch.isfinite(l_ic)

    t_bc = torch.rand(5, 1)
    l_bc = loss_bc(cfg, model, t_bc)
    assert torch.isfinite(l_bc)


def test_loss_physics_inverse_is_finite_and_differentiable():
    cfg = small_cfg()
    model = InverseFCNet(cfg)
    t_col = torch.rand(5, 1, requires_grad=True)
    x_col = torch.rand(5, 1, requires_grad=True)
    l_phys = loss_physics_inverse(model, t_col, x_col)
    assert torch.isfinite(l_phys)
    l_phys.backward()
    assert model._nu_raw.grad is not None


def test_predict_from_state_matches_direct_predict():
    cfg = small_cfg()
    model = FCNet(cfg)
    t = np.linspace(0.1, cfg.t_dom, 10)
    x = np.linspace(0, cfg.L, 10)

    state_dict = model.state_dict()
    u_direct = predict(model, t, x)
    u_restored = predict_from_state(state_dict, t, x, cfg)

    assert np.allclose(u_direct, u_restored)
