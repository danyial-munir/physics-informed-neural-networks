"""
Neural network architecture and inference helpers for the Fisher-KPP PINN.

Contains:
    FCNet         -- fully-connected forward network with Tanh activations
    InverseFCNet  -- forward network plus learnable D, r parameters
    predict       -- predict u-values for arrays of (t, x)
    predict_from_state -- restore a snapshot and predict
"""

import torch
import torch.nn as nn
import numpy as np
from config import Config


class FCNet(nn.Module):
    """
    Fully-connected network: 2 -> [hidden]*n_layers -> 1
    Tanh activations throughout.

    Tanh is mandatory (not ReLU) because we differentiate the network
    output twice via autograd for the physics residual.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        layers = [nn.Linear(2, cfg.hidden), nn.Tanh(), nn.Dropout(cfg.dropout_rate)]
        for _ in range(cfg.n_layers - 1):
            layers += [
                nn.Linear(cfg.hidden, cfg.hidden),
                nn.Tanh(),
                nn.Dropout(cfg.dropout_rate),
            ]
        layers += [nn.Linear(cfg.hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(torch.cat([t, x], dim=1))

    def param_count(self) -> int:
        """Return parameter count."""
        return sum(p.numel() for p in self.parameters())


class InverseFCNet(nn.Module):
    """
    Same backbone as FCNet, plus two learnable raw parameters exposed as
    positive D_hat, r_hat via softplus (softplus-inverse initialised).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        layers = [nn.Linear(2, cfg.hidden), nn.Tanh(), nn.Dropout(cfg.dropout_rate)]
        for _ in range(cfg.n_layers - 1):
            layers += [
                nn.Linear(cfg.hidden, cfg.hidden),
                nn.Tanh(),
                nn.Dropout(cfg.dropout_rate),
            ]
        layers += [nn.Linear(cfg.hidden, 1)]
        self.net = nn.Sequential(*layers)

        D_init = np.random.uniform(0.5, 2.0)
        D_init = float(np.log(np.exp(D_init) - 1.0))  # softplus inverse
        self._D_raw = nn.Parameter(torch.tensor([D_init], requires_grad=True))

        r_init = np.random.uniform(0.5, 2.0)
        r_init = float(np.log(np.exp(r_init) - 1.0))
        self._r_raw = nn.Parameter(torch.tensor([r_init], requires_grad=True))

    @property
    def D_hat(self):
        """Positive diffusion coefficient."""
        return torch.nn.functional.softplus(self._D_raw)

    @property
    def r_hat(self):
        """Positive growth rate."""
        return torch.nn.functional.softplus(self._r_raw)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(torch.cat([t, x], dim=1))

    def param_count(self) -> int:
        """Return parameter count."""
        return sum(p.numel() for p in self.parameters())


def predict(model: nn.Module, t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Run inference on CPU. The model must already be on CPU."""
    model.eval()
    t_t = torch.tensor(t, dtype=torch.float32).unsqueeze(1)
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        return model(t_t, x_t).squeeze().numpy()


def predict_from_state(
    state_dict: dict, t: np.ndarray, x: np.ndarray, cfg: Config
) -> np.ndarray:
    """Restore a CPU snapshot and predict."""
    tmp = FCNet(cfg)
    tmp.load_state_dict(state_dict)
    tmp.eval()
    return predict(tmp, t, x)
