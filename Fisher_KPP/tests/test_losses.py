import torch

from config import Config
from model import FCNet, InverseFCNet
from losses import loss_data, loss_physics, loss_physics_inverse, loss_ic, loss_bc


def small_cfg(**overrides) -> Config:
    defaults = dict(n_x=100, T=1.0, n_t=20, n_obs=5, n_col=10, n_bc=5, n_grid_val=5)
    defaults.update(overrides)
    return Config(**defaults)


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
    assert model._D_raw.grad is not None
    assert model._r_raw.grad is not None


def test_loss_bc_is_non_negative():
    cfg = small_cfg()
    model = FCNet(cfg)
    t_bc = torch.rand(5, 1)
    l_bc = loss_bc(cfg, model, t_bc)
    assert l_bc.item() >= 0.0
