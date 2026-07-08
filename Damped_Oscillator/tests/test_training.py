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
