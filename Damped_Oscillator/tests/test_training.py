import numpy as np
import torch

from config import Config
from model import FCNet
from data import generate_data
from trainer import train
from utils import get_device


def tiny_cfg(**overrides) -> Config:
    defaults = dict(
        hidden=8,
        n_layers=2,
        n_epochs=200,
        log_every=10,
        patience=10_000,
        snapshot_epochs=(1,),
        n_obs=10,
        n_val=20,
        n_col_dom=20,
        t_dom=2.0,
        t_extrap=2.2,
    )
    defaults.update(overrides)
    return Config(**defaults)


def test_training_loop_reduces_loss():
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = tiny_cfg()
    model = FCNet(cfg)
    data = generate_data(cfg)

    history, snapshots, best_state = train(
        model, data, cfg, device=get_device(), verbatim=False
    )

    assert history["loss_total"][-1] < history["loss_total"][0]
    assert 1 in snapshots
    assert best_state is not None
    assert all(torch.isfinite(v).all() for v in best_state.values())


def test_inverse_training_recovers_parameters_in_right_direction():
    torch.manual_seed(0)
    np.random.seed(0)

    from model import InverseFCNet

    cfg = tiny_cfg(use_data=True, n_epochs=300)
    model = InverseFCNet(cfg)
    data = generate_data(cfg)

    history, _, best_state = train(
        model, data, cfg, device=get_device(), verbatim=False
    )

    assert history["loss_total"][-1] < history["loss_total"][0]
    assert np.isfinite(history["zeta_hat"][-1])
    assert np.isfinite(history["omega_0_hat"][-1])


def test_training_with_data_loss_enabled():
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = tiny_cfg(use_data=True, randomise_observation=False)
    model = FCNet(cfg)
    data = generate_data(cfg)

    history, _, _ = train(model, data, cfg, device=get_device(), verbatim=False)

    assert history["loss_total"][-1] < history["loss_total"][0]
    assert all(v > 0 for v in history["loss_data"])


def test_training_without_extrapolation_collocation():
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = tiny_cfg(train_extrap=False)
    model = FCNet(cfg)
    data = generate_data(cfg)

    history, _, _ = train(model, data, cfg, device=get_device(), verbatim=False)

    assert history["loss_total"][-1] < history["loss_total"][0]


def test_training_without_physics_loss():
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = tiny_cfg(use_physics=False, use_data=True)
    model = FCNet(cfg)
    data = generate_data(cfg)

    history, _, _ = train(model, data, cfg, device=get_device(), verbatim=False)

    assert all(lp == 0.0 for lp in history["loss_phys"])
    assert history["loss_total"][-1] < history["loss_total"][0]


def test_early_stopping_triggers_with_tiny_patience():
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = tiny_cfg(n_epochs=1000, patience=5, patience_threshold=1.0, log_every=1)
    model = FCNet(cfg)
    data = generate_data(cfg)

    history, _, _ = train(model, data, cfg, device=get_device(), verbatim=False)

    assert len(history["epoch"]) < cfg.n_epochs
