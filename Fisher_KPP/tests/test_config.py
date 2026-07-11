import numpy as np
import torch

from config import Config
from utils import get_device, to_tensor, rmse


def test_config_defaults():
    cfg = Config()
    assert cfg.D == 1.0
    assert cfg.r == 1.0
    assert cfg.eps == 0.5
    assert cfg.L == 20.0
    assert cfg.T == 6.0


def test_config_delta_x_and_delta_t_properties():
    cfg = Config(L=20.0, n_x=400, T=6.0, n_t=600)
    assert cfg.delta_x == 2 * cfg.L / cfg.n_x
    assert cfg.delta_t == cfg.T / cfg.n_t


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)


def test_to_tensor_shapes_and_grad():
    arr = np.array([1.0, 2.0, 3.0])
    t = to_tensor(arr, requires_grad=True)
    assert t.shape == (3, 1)
    assert t.requires_grad is True


def test_rmse_zero_for_identical_arrays():
    a = np.array([1.0, 2.0, 3.0])
    assert rmse(a, a) == 0.0
