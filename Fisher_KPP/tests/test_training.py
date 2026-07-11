import numpy as np
import pytest
import torch
import optuna

from config import Config
from model import FCNet, InverseFCNet
from data import generate_data
from trainer import train
from utils import get_device


def tiny_cfg(**overrides) -> Config:
    defaults = dict(
        hidden=8,
        n_layers=2,
        n_epochs=100,
        log_every=10,
        patience=10_000,
        snapshot_epochs=(1,),
        n_x=100,
        T=1.0,
        n_t=20,
        n_obs=10,
        n_col=20,
        n_bc=10,
        n_grid_val=10,
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


def test_inverse_training_recovers_positive_finite_parameters():
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = tiny_cfg(use_data=True, n_epochs=150)
    model = InverseFCNet(cfg)
    data = generate_data(cfg)

    history, _, _ = train(model, data, cfg, device=get_device(), verbatim=False)

    assert history["loss_total"][-1] < history["loss_total"][0]
    assert np.isfinite(history["D_hat"][-1])
    assert np.isfinite(history["r_hat"][-1])
    assert history["D_hat"][-1] > 0
    assert history["r_hat"][-1] > 0


def test_training_without_physics_or_bc_loss():
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = tiny_cfg(use_physics=False, use_bc=False, use_data=True)
    model = FCNet(cfg)
    data = generate_data(cfg)

    history, _, _ = train(model, data, cfg, device=get_device(), verbatim=False)

    assert all(lp == 0.0 for lp in history["loss_phys"])
    assert all(lb == 0.0 for lb in history["loss_bc"])
    assert history["loss_total"][-1] < history["loss_total"][0]


def test_early_stopping_triggers_with_tiny_patience():
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = tiny_cfg(n_epochs=1000, patience=5, patience_threshold=1.0, log_every=1)
    model = FCNet(cfg)
    data = generate_data(cfg)

    history, _, _ = train(model, data, cfg, device=get_device(), verbatim=False)

    assert len(history["epoch"]) < cfg.n_epochs


def test_train_raises_trial_pruned_when_trial_requests_prune():
    torch.manual_seed(0)
    np.random.seed(0)

    class AlwaysPruneTrial:
        def report(self, value, step):
            pass

        def should_prune(self):
            return True

    cfg = tiny_cfg(log_every=1)
    model = FCNet(cfg)
    data = generate_data(cfg)

    with pytest.raises(optuna.exceptions.TrialPruned):
        train(model, data, cfg, device=get_device(), verbatim=False, optuna_trial=AlwaysPruneTrial())


def test_inverse_training_then_plotting_does_not_crash(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from utils import save_model
    from plot import save_plots_from_file

    torch.manual_seed(0)
    np.random.seed(0)

    cfg = tiny_cfg(use_data=True, n_epochs=20, log_every=5)
    model = InverseFCNet(cfg)
    data = generate_data(cfg)

    history, snapshots, best_state = train(
        model, data, cfg, device=get_device(), verbatim=False
    )

    save_model(best_state, history, snapshots, cfg, tmp_path)
    save_plots_from_file(tmp_path, verbatim=False)

    assert (tmp_path / "solution_heatmap.png").exists()
    assert (tmp_path / "snapshots.png").exists()
    assert (tmp_path / "losses.png").exists()
    assert (tmp_path / "parameter_convergence.png").exists()
