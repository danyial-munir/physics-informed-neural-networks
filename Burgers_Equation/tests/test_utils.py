import json

import numpy as np
import torch

from config import Config
from model import FCNet
from utils import rmse, save_model, load_model, load_best_cfg


def test_rmse_zero_for_identical_arrays():
    a = np.array([1.0, 2.0, 3.0])
    assert rmse(a, a) == 0.0


def test_rmse_known_value():
    pred = np.array([0.0, 0.0])
    true = np.array([3.0, 4.0])
    assert np.isclose(rmse(pred, true), np.sqrt((9 + 16) / 2))


def test_save_and_load_model_roundtrip(tmp_path):
    cfg = Config(hidden=4, n_layers=1, n_x=50)
    model = FCNet(cfg)
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    history = {"epoch": [1, 2], "loss_total": [1.0, 0.5]}
    snapshots = {1: best_state}

    save_model(best_state, history, snapshots, cfg, tmp_path, verbatim=False)

    loaded_model, loaded_history, loaded_snapshots, loaded_cfg = load_model(
        FCNet(cfg), tmp_path
    )

    for k, v in best_state.items():
        assert torch.equal(v, loaded_model.state_dict()[k])
    assert loaded_history["loss_total"] == history["loss_total"]
    assert loaded_cfg.hidden == cfg.hidden


def test_load_best_cfg_reads_optuna_json(tmp_path):
    json_path = tmp_path / "best.json"
    json_path.write_text(json.dumps({"best_params": {"hidden": 32, "n_layers": 3}}))

    cfg = load_best_cfg(str(json_path))
    assert cfg.hidden == 32
    assert cfg.n_layers == 3


def test_load_best_cfg_maps_regime_to_ic(tmp_path):
    json_path = tmp_path / "best.json"
    json_path.write_text(
        json.dumps({"best_params": {"hidden": 16, "n_layers": 2}, "regime": "N_wave"})
    )

    cfg = load_best_cfg(str(json_path))
    assert cfg.ic == "N_wave"
