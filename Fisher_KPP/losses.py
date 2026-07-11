"""
Loss functions for the Fisher-KPP PINN.

Contains:
    loss_data            -- MSE between predictions and noisy observations
    loss_physics          -- mean squared PDE residual (known D, r)
    loss_physics_inverse   -- mean squared PDE residual (learned D_hat, r_hat)
    loss_ic                -- squared error on the initial condition
    loss_bc                -- squared error on the Dirichlet boundary values
"""

import torch
import torch.nn as nn
from config import Config


def loss_data(u_pred: torch.Tensor, u_obs_t: torch.Tensor) -> torch.Tensor:
    """Mean squared error between predictions and observations."""
    return torch.mean((u_pred - u_obs_t) ** 2)


def _physics_residual(
    model: nn.Module, t_col_t: torch.Tensor, x_col_t: torch.Tensor, D, r
) -> torch.Tensor:
    """r(t,x) = u_t - D*u_xx - r*u*(1-u)."""
    u_hat = model(t_col_t, x_col_t)

    u_t = torch.autograd.grad(
        u_hat, t_col_t, grad_outputs=torch.ones_like(u_hat), create_graph=True
    )[0]

    u_x = torch.autograd.grad(
        u_hat, x_col_t, grad_outputs=torch.ones_like(u_hat), create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x, x_col_t, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]

    return u_t - D * u_xx - r * u_hat * (1.0 - u_hat)


def loss_physics(
    cfg: Config, model: nn.Module, t_col_t: torch.Tensor, x_col_t: torch.Tensor
) -> torch.Tensor:
    """L_physics = mean( residual^2 ) over collocation points, known D, r."""
    residual = _physics_residual(model, t_col_t, x_col_t, cfg.D, cfg.r)
    return torch.mean(residual**2)


def loss_physics_inverse(
    model: nn.Module, t_col_t: torch.Tensor, x_col_t: torch.Tensor
) -> torch.Tensor:
    """L_physics = mean( residual^2 ) over collocation points, learned D_hat, r_hat."""
    residual = _physics_residual(model, t_col_t, x_col_t, model.D_hat, model.r_hat)
    return torch.mean(residual**2)


def loss_ic(
    model: nn.Module,
    t_ic_t: torch.Tensor,
    x_ic_t: torch.Tensor,
    u_ic_true: torch.Tensor,
) -> torch.Tensor:
    """L_ic = mean( (u_hat(0,x) - u_0(x))^2 )."""
    u_hat_t0 = model(t_ic_t, x_ic_t)
    return torch.mean((u_hat_t0 - u_ic_true) ** 2)


def loss_bc(cfg: Config, model: nn.Module, t_bc_t: torch.Tensor) -> torch.Tensor:
    """L_bc = mean( (u_hat(t,-L) - 1)^2 + (u_hat(t,L) - 0)^2 )."""
    x_left = torch.full_like(t_bc_t, -cfg.L)
    x_right = torch.full_like(t_bc_t, cfg.L)

    u_hat_left = model(t_bc_t, x_left)
    u_hat_right = model(t_bc_t, x_right)

    return torch.mean((u_hat_left - 1.0) ** 2 + (u_hat_right - 0.0) ** 2)
