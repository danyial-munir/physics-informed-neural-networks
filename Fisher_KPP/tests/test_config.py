import json

import numpy as np
import torch

from config import Config
from utils import get_device, to_tensor, rmse, load_best_cfg


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


def test_t_train_defaults_to_T_when_unset():
    cfg = Config(T=6.0)
    assert cfg.t_train == 6.0


def test_t_train_preserves_explicit_value():
    cfg = Config(T=6.0, t_train=3.0)
    assert cfg.t_train == 3.0


def test_train_extrap_defaults_true():
    cfg = Config()
    assert cfg.train_extrap is True


def test_load_best_cfg_reads_params_from_json(tmp_path):
    payload = {"best_val": 0.01, "best_params": {"D": 0.5, "r": 2.0, "hidden": 32}}
    json_path = tmp_path / "best_params.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = load_best_cfg(str(json_path))
    assert cfg.D == 0.5
    assert cfg.r == 2.0
    assert cfg.hidden == 32
