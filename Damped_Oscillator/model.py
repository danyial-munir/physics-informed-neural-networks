"""
Neural network architecture and inference helpers for the damped
spring-mass PINN.

Contains:
    FCNet        -- fully-connected network with Tanh activations
    predict      -- predict y-values for array of t-values
    predict_from_state -- restore a snapshot and predict

Author          : Danyial Munir
Email           : danyial.munir@studio.unibo.it
Last modified   : 12/05/2026
"""

import torch
import torch.nn as nn
import numpy as np
from config import Config


class FCNet(nn.Module):
    """
    Fully-connected network: 1 -> [hidden]*n_layers -> 1
    Tanh activations throughout.

    Tanh is mandatory (not ReLU) because we differentiate the network output
    twice via autograd.  ReLU has zero second derivative everywhere, which
    would make the physics residual carry no gradient signal.

    The model is instantiated on CPU and moved to DEVICE via .to(DEVICE)
    after construction (see training section).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        layers = [nn.Linear(1, cfg.hidden), nn.Tanh(), nn.Dropout(cfg.dropout_rate)]
        for _ in range(cfg.n_layers - 1):
            layers += [
                nn.Linear(cfg.hidden, cfg.hidden),
                nn.Tanh(),
                nn.Dropout(cfg.dropout_rate),
            ]
        layers += [nn.Linear(cfg.hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass throught the network."""
        return self.net(t)

    def param_count(self) -> int:
        """Return parameter count."""
        return sum(p.numel() for p in self.parameters())


class InverseFCNet(nn.Module):
    """
    Inverse Fully-connected network: 1 -> [hidden]*n_layers -> 1
    Tanh activations throughout. Can be used to predict equation parameters

    Tanh is mandatory (not ReLU) because we differentiate the network output
    twice via autograd.  ReLU has zero second derivative everywhere, which
    would make the physics residual carry no gradient signal.

    The model is instantiated on CPU and moved to DEVICE via .to(DEVICE)
    after construction (see training section).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        layers = [nn.Linear(1, cfg.hidden), nn.Tanh(), nn.Dropout(cfg.dropout_rate)]
        for _ in range(cfg.n_layers - 1):
            layers += [
                nn.Linear(cfg.hidden, cfg.hidden),
                nn.Tanh(),
                nn.Dropout(cfg.dropout_rate),
            ]
        layers += [nn.Linear(cfg.hidden, 1)]
        self.net = nn.Sequential(*layers)

        # zeta_init = np.random.uniform(0.1, 2)   # random start
        # omega_0_init = np.random.uniform(3, 5)
        self.zeta_hat = nn.Parameter(torch.tensor([0.1], requires_grad=True))
        self.omega_0_hat = nn.Parameter(torch.tensor([0.5], requires_grad=True))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass throught the network."""
        return self.net(t)

    def param_count(self) -> int:
        """Return parameter count."""
        return sum(p.numel() for p in self.parameters())


def predict(model: nn.Module, t: np.ndarray) -> np.ndarray:
    """
    Run inference on CPU.  The model must already be on CPU.
    No gradients needed -- torch.no_grad() saves memory and time.
    """
    model.eval()
    t_t = torch.tensor(t, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        return model(t_t).squeeze().numpy()


def predict_from_state(state_dict: dict, t: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Restore a CPU snapshot and predict.
    Automatically detects forward vs inverse model from state dict keys.
    """
    inverse = any("zeta_hat" in k or "omega_0_hat" in k for k in state_dict.keys())
    tmp = InverseFCNet(cfg) if inverse else FCNet(cfg)
    tmp.load_state_dict(state_dict)
    tmp.eval()
    return predict(tmp, t)
