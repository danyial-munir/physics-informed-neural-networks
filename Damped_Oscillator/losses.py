"""
Loss functions for the damped spring-mass PINN.

Each function takes a model and the relevant tensors
as arguments.
Contains:
    loss_data    -- MSE between predictions and noisy observations
    loss_physics -- mean squared ODE residual at collocation points
    loss_ic      -- squared error on initial position and velocity

Author          : Danyial Munir
Email           : danyial.munir@studio.unibo.it
Last modified   : 12/05/2026
"""

import torch
from torch import nn
from config import Config


def loss_data(y_pred: torch.Tensor, y_obs_t: torch.Tensor) -> torch.Tensor:
    """Compute the mean squared error between predictions and observations."""
    l_data = torch.mean((y_pred - y_obs_t) ** 2)
    return l_data


def loss_physics(cfg: Config, model: nn.Module, t_col_t: torch.Tensor) -> torch.Tensor:
    """
    L_physics = mean( r(t_i)^2 )  over collocation points.

    r(t) = m*y_hat''(t) + c*y_hat'(t) + k*y_hat(t)

    Both derivatives are obtained via automatic differentiation through the
    network.  create_graph=True on the first call keeps the computation graph
    alive so the second differentiation can flow back through it.

    The [0] unpacks the single-element tuple returned by autograd.grad
    (the API always returns a tuple, one entry per input tensor).
    """
    y_hat = model(t_col_t)

    # first derivative  dy/dt
    dy = torch.autograd.grad(
        y_hat, t_col_t, grad_outputs=torch.ones_like(y_hat), create_graph=True
    )[0]

    # second derivative  d2y/dt2
    d2y = torch.autograd.grad(
        dy, t_col_t, grad_outputs=torch.ones_like(dy), create_graph=True
    )[0]

    residual = cfg.m * d2y + cfg.c * dy + cfg.k * y_hat

    return torch.mean(residual**2)


def loss_physics_inverse(model: nn.Module, t_col_t: torch.Tensor) -> torch.Tensor:
    """
    L_physics = mean( r(t_i)^2 )  over collocation points.

    r(t) = m*y_hat''(t) + c*y_hat'(t) + k*y_hat(t)

    Both derivatives are obtained via automatic differentiation through the
    network.  create_graph=True on the first call keeps the computation graph
    alive so the second differentiation can flow back through it.

    The [0] unpacks the single-element tuple returned by autograd.grad
    (the API always returns a tuple, one entry per input tensor).
    """
    y_hat = model(t_col_t)

    # first derivative  dy/dt
    dy = torch.autograd.grad(
        y_hat, t_col_t, grad_outputs=torch.ones_like(y_hat), create_graph=True
    )[0]

    # second derivative  d2y/dt2
    d2y = torch.autograd.grad(
        dy, t_col_t, grad_outputs=torch.ones_like(dy), create_graph=True
    )[0]

    residual = (
        d2y + 2 * model.zeta_hat * model.omega_0_hat * dy + model.omega_0_hat**2 * y_hat
    )

    return torch.mean(residual**2)


def loss_ic(cfg: Config, model: nn.Module, t_ic_t: torch.Tensor) -> torch.Tensor:
    """
    L_ic = ( y_hat(0) - Y0 )^2  +  ( y_hat'(0) - DY0 )^2.

    Both initial conditions are enforced explicitly.
    y_hat'(0) is obtained via autograd because initial velocity
    is never directly observed in the data.

    LAMBDA_IC >> LAMBDA_PHYS: an error at t=0 compounds over the
    entire trajectory, so this constraint must be tight.
    """
    y_hat_0 = model(t_ic_t)

    dy_0 = torch.autograd.grad(
        y_hat_0, t_ic_t, grad_outputs=torch.ones_like(y_hat_0), create_graph=True
    )[0]

    return torch.mean((y_hat_0 - cfg.y0) ** 2 + (dy_0 - cfg.dy0) ** 2)
