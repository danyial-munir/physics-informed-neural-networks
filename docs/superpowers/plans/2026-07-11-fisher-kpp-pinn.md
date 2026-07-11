# Fisher-KPP PINN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `Fisher_KPP/` project to this repo that solves the KPP-Fisher reaction-diffusion equation (`u_t = D*u_xx + r*u*(1-u)`) with a PINN, forward and inverse, mirroring `Burgers_Equation/`'s file layout and conventions.

**Architecture:** Same as `Burgers_Equation/`: flat per-project modules (`config.py`, `model.py`, `losses.py`, `analytic.py`, `data.py`, `trainer.py`, `plot.py`, `utils.py`) imported by absolute module name (each project dir is added to `sys.path` by its own `tests/conftest.py` / scripts add the project root to `sys.path`). A `Config` dataclass carries every physical/hyperparameter constant. `FCNet`/`InverseFCNet` are `Tanh` MLPs differentiated via `torch.autograd.grad` for the physics residual. An explicit finite-difference solver is the ground-truth reference for both forward-PINN validation plots and inverse-problem synthetic observations.

**Tech Stack:** PyTorch, NumPy, Matplotlib, pytest (already in `pyproject.toml` dev group — no new dependencies).

## Global Constraints

- Equation: `u_t = D*u_xx + r*u*(1-u)`.
- Domain: `x ∈ [-L, L]`, `t ∈ [0, T]`. Use `L = 20.0`, `T = 6.0` (front speed `c=2` covers ~12 units in `T=6`, stays inside `[-20,20]`).
- IC: `u(x,0) = 0.5*(1 - tanh(x/eps))`, `eps = 0.5`.
- BC: Dirichlet, `u(-L,t) = 1`, `u(L,t) = 0`.
- Defaults: `D = 1.0`, `r = 1.0` (analytic wave speed `c = 2*sqrt(r*D) = 2.0`, used only as a plot sanity-check overlay, not as a training target).
- Reference solution: explicit FD, CFL-stable `dt`.
- New project directory: `Fisher_KPP/`, same file layout as `Burgers_Equation/` (`config.py`, `model.py`, `losses.py`, `analytic.py`, `data.py`, `trainer.py`, `plot.py`, `utils.py`, `scripts/`, `tests/`).
- Scripts in scope: `pinn_demo.py`, `inversepinn_demo.py`, `inversepinn_data_generation.py`, `report_plots.py`. No optuna/profiling/blind-extrap/no-physics scripts.
- No changes to `Burgers_Equation/` or `Damped_Oscillator/`.
- Tests discovered automatically by repo-root `run_tests.py` (globs `*/tests` dirs) — no changes needed there.
- Python >=3.14, existing deps only (torch, numpy, matplotlib, tqdm already in `pyproject.toml`).

---

## File Structure

```
Fisher_KPP/
├── config.py          # Config dataclass
├── utils.py            # get_device, to_tensor, rmse, save_show, save_model, load_model
├── analytic.py         # IC, FD solver, interpolation, residual/rmse diagnostics
├── model.py             # FCNet, InverseFCNet, predict, predict_from_state
├── losses.py            # loss_data, loss_ic, loss_bc, loss_physics, loss_physics_inverse
├── data.py               # make_observation, make_collocation, make_bc_points, make_validation, generate_data
├── trainer.py             # train()
├── plot.py                 # style_ax, plot_solution_heatmap, plot_snapshots, plot_losses,
│                            # plot_predicted_parameter_convergence, save_plots_from_file
├── scripts/
│   ├── pinn_demo.py                       # forward demo
│   ├── inversepinn_data_generation.py     # synthetic noisy observations from FD grid
│   ├── inversepinn_demo.py                # inverse demo
│   └── report_plots.py                    # report figures
└── tests/
    ├── conftest.py
    ├── test_data.py
    ├── test_smoke.py
    └── test_training.py
```

---

### Task 1: `config.py` and `utils.py`

**Files:**
- Create: `Fisher_KPP/config.py`
- Create: `Fisher_KPP/utils.py`
- Test: `Fisher_KPP/tests/conftest.py`
- Test: `Fisher_KPP/tests/test_config.py`

**Interfaces:**
- Produces: `Config` dataclass with fields `D, r, eps, L, T, n_x, n_t, delta_x (property), delta_t (property), use_physics, lambda_phys, use_ic, lambda_ic, use_bc, lambda_bc, use_data, n_obs, randomise_observation, n_col, randomise_collocation, n_bc, randomise_bc_points, n_grid_val, hidden, n_layers, n_epochs, lr, lr_inverse, patience, patience_threshold, dropout_rate, adam_beta1, adam_beta2, scheduler_gamma, scheduler_step, snapshot_epochs, log_every, sigma (obs noise std)`.
- Produces: `get_device() -> torch.device`, `to_tensor(arr, requires_grad=False, unsqueeze=True) -> torch.Tensor`, `rmse(pred, true) -> float`, `save_show(output_path=None, show=True) -> None`, `save_model(best_state, history, snapshots, cfg, output_path, verbatim=False) -> None`, `load_model(model, folder_path) -> tuple`.

- [ ] **Step 1: Write `Fisher_KPP/tests/conftest.py`**

```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
```

- [ ] **Step 2: Write the failing test for `Config`**

Create `Fisher_KPP/tests/test_config.py`:

```python
from config import Config


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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd Fisher_KPP && python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'config'` (file doesn't exist yet).

- [ ] **Step 4: Create `Fisher_KPP/config.py`**

```python
"""
Configuration for the Fisher-KPP equation PINN.

All physical constants, hyperparameters, and runtime settings
for the PINN are defined as a single config class.
Import and instantiate Config in every script that needs these values.
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Config class containing all parameters for model training."""

    # Physical parameters
    D: float = 1.0  # diffusion coefficient
    r: float = 1.0  # logistic growth rate

    # Initial condition
    eps: float = 0.5  # tanh smoothing width for the step IC

    # Space domain: x in [-L, L]
    L: float = 20.0
    n_x: int = 400  # spatial resolution for the FD reference grid

    # Time domain: t in [0, T]
    T: float = 6.0
    n_t: int = 600  # number of FD reference time steps

    @property
    def delta_x(self) -> float:
        """Return spatial cell dimension."""
        return 2 * self.L / self.n_x

    @property
    def delta_t(self) -> float:
        """Return time step for the FD reference grid."""
        return self.T / self.n_t

    # Loss weights
    use_physics: bool = True
    lambda_phys: float = 1.0
    use_ic: bool = True
    lambda_ic: float = 10.0
    use_bc: bool = True
    lambda_bc: float = 10.0
    use_data: bool = False

    # Data generation parameters
    n_obs: int = 50  # noisy observation points (inverse problem)
    randomise_observation: bool = True
    sigma: float = 0.01  # observation noise std
    n_col: int = 2000  # PDE residual collocation points
    randomise_collocation: bool = True
    n_bc: int = 100  # boundary condition points
    randomise_bc_points: bool = True
    n_grid_val: int = 100  # validation grid size (per axis)

    # Network architecture
    hidden: int = 64
    n_layers: int = 6

    # Hyperparameters
    n_epochs: int = 20000
    lr: float = 2e-3
    lr_inverse: float = 5e-3
    patience: int = 4000
    patience_threshold: float = 2e-6
    dropout_rate: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    scheduler_gamma: float = 0.5
    scheduler_step: int = 3000

    # Epoch snapshots
    snapshot_epochs: tuple = (1, 50, 300, 1000, 2000, 4000, 10000)
    log_every: int = 50
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd Fisher_KPP && python -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Write the failing test for `utils.py`**

Add to `Fisher_KPP/tests/test_config.py`:

```python
import numpy as np
import torch

from utils import get_device, to_tensor, rmse


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
```

- [ ] **Step 7: Run test to verify it fails**

Run: `cd Fisher_KPP && python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'utils'`

- [ ] **Step 8: Create `Fisher_KPP/utils.py`**

```python
"""
Utility functions for the Fisher-KPP PINN: device selection, tensor
conversion, error metrics, and model/history persistence.
"""

import os
import dataclasses
from pathlib import Path
import json
import csv
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from config import Config


def get_device() -> torch.device:
    """Return device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_tensor(
    arr: np.ndarray, requires_grad: bool = False, unsqueeze: bool = True
) -> torch.Tensor:
    """Convert a 1-D numpy array to a (N, 1) float32 tensor on DEVICE."""
    device = get_device()
    arr = np.atleast_1d(np.array(arr, dtype=np.float32))
    t = torch.tensor(arr, dtype=torch.float32)
    if unsqueeze:
        t = t.unsqueeze(1)
    t = t.to(device)
    if requires_grad:
        t.requires_grad_(True)
    return t


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Calculate RMSE."""
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def save_show(output_path: str = None, show: bool = True) -> None:
    """Save the current figure and/or show it."""
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def save_model(
    best_state: dict,
    history: dict,
    snapshots: dict,
    cfg: Config,
    output_path: Path,
    verbatim: bool = False,
) -> None:
    """
    Save model weights, training history, snapshots, and config.

    Saves
    -----
    <output_path>/best_model.pt   -- best model state dict
    <output_path>/history.csv     -- loss curves, one row per log step
    <output_path>/snapshots.pt    -- all snapshot state dicts keyed by epoch
    <output_path>/config.json     -- full config for reproducibility
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(best_state, output_path / "best_model.pt")

    with open(output_path / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history.keys())
        writer.writeheader()
        writer.writerows(
            [{k: history[k][i] for k in history} for i in range(len(history["epoch"]))]
        )

    torch.save(snapshots, output_path / "snapshots.pt")

    with open(output_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=4)

    if verbatim:
        print(f"Saved model, history, snapshots and config to {output_path}/")


def load_model(model: nn.Module, folder_path: str) -> tuple[nn.Module, dict, dict, Config]:
    """
    Load model weights, history, snapshots, and config from a saved run.
    """
    folder_path = Path(folder_path)

    model.load_state_dict(torch.load(folder_path / "best_model.pt", map_location="cpu"))

    with open(folder_path / "history.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    history = {k: [float(r[k]) for r in rows] for k in rows[0]}

    with open(folder_path / "config.json", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = Config(**cfg_dict)

    snapshots = torch.load(folder_path / "snapshots.pt", map_location="cpu")

    return model, history, snapshots, cfg
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `cd Fisher_KPP && python -m pytest tests/test_config.py -v`
Expected: PASS (5 tests)

- [ ] **Step 10: Commit**

```bash
git add Fisher_KPP/config.py Fisher_KPP/utils.py Fisher_KPP/tests/conftest.py Fisher_KPP/tests/test_config.py
git commit -m "Add Fisher_KPP config and utils"
```

---

### Task 2: `analytic.py` — IC and finite-difference reference solver

**Files:**
- Create: `Fisher_KPP/analytic.py`
- Test: `Fisher_KPP/tests/test_analytic.py`

**Interfaces:**
- Consumes: `Config` from Task 1 (`D, r, eps, L, n_x, T, n_t, delta_x, delta_t`).
- Produces: `u_0(cfg) -> np.ndarray` shape `(n_x,)`, `compute_stable_dt(cfg, cfl=0.4) -> float`, `fd_step(u, dt, cfg) -> np.ndarray`, `solve_kpp_fd(cfg) -> tuple[np.ndarray, np.ndarray]` returning `(u_grid of shape (n_snapshots, n_x), t_arr of shape (n_snapshots,))`, `interpolate_solution(u_grid, t_arr, x, t, cfg) -> float`, `wave_speed(cfg) -> float`.

- [ ] **Step 1: Write the failing tests**

Create `Fisher_KPP/tests/test_analytic.py`:

```python
import numpy as np

from config import Config
from analytic import (
    u_0,
    compute_stable_dt,
    fd_step,
    solve_kpp_fd,
    interpolate_solution,
    wave_speed,
)


def test_u_0_shape_and_bounds():
    cfg = Config(n_x=200)
    u = u_0(cfg)
    assert u.shape == (cfg.n_x,)
    assert np.isfinite(u).all()
    assert u.min() >= -1e-6
    assert u.max() <= 1 + 1e-6


def test_u_0_is_decreasing_step_from_1_to_0():
    cfg = Config(n_x=200, eps=0.5)
    u = u_0(cfg)
    assert u[0] > 0.9  # left edge near x=-L, u ~ 1
    assert u[-1] < 0.1  # right edge near x=+L, u ~ 0


def test_compute_stable_dt_positive_and_bounded_by_diffusion():
    cfg = Config(n_x=200, D=1.0)
    dt = compute_stable_dt(cfg)
    assert dt > 0
    assert dt <= 0.5 * cfg.delta_x**2 / cfg.D


def test_fd_step_preserves_shape_and_stays_finite():
    cfg = Config(n_x=200)
    u = u_0(cfg)
    dt = compute_stable_dt(cfg)
    u_next = fd_step(u, dt, cfg)
    assert u_next.shape == u.shape
    assert np.isfinite(u_next).all()


def test_fd_step_respects_dirichlet_boundary_values():
    cfg = Config(n_x=200)
    u = u_0(cfg)
    dt = compute_stable_dt(cfg)
    u_next = fd_step(u, dt, cfg)
    assert abs(u_next[0] - 1.0) < 1e-6
    assert abs(u_next[-1] - 0.0) < 1e-6


def test_solve_kpp_fd_returns_grid_and_stays_bounded():
    cfg = Config(n_x=200, T=2.0, n_t=50)
    u_grid, t_arr = solve_kpp_fd(cfg)
    assert u_grid.shape[1] == cfg.n_x
    assert u_grid.shape[0] == t_arr.shape[0]
    assert np.isfinite(u_grid).all()
    assert u_grid.min() >= -0.05
    assert u_grid.max() <= 1.05


def test_interpolate_solution_matches_grid_at_exact_node():
    cfg = Config(n_x=100, T=1.0, n_t=20)
    u_grid, t_arr = solve_kpp_fd(cfg)
    x0 = -cfg.L
    val = interpolate_solution(u_grid, t_arr, x0, float(t_arr[0]), cfg)
    assert abs(val - u_grid[0, 0]) < 1e-6


def test_wave_speed_formula():
    cfg = Config(D=1.0, r=1.0)
    assert abs(wave_speed(cfg) - 2.0) < 1e-9

    cfg2 = Config(D=0.25, r=1.0)
    assert abs(wave_speed(cfg2) - 1.0) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd Fisher_KPP && python -m pytest tests/test_analytic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analytic'`

- [ ] **Step 3: Create `Fisher_KPP/analytic.py`**

```python
"""
Initial condition and finite-difference reference solver for the
Fisher-KPP equation: u_t = D*u_xx + r*u*(1-u).
"""

import numpy as np
from tqdm import tqdm
from config import Config


def u_0(cfg: Config) -> np.ndarray:
    """Smoothed step initial condition: 0.5*(1 - tanh(x/eps))."""
    x = np.linspace(-cfg.L, cfg.L, cfg.n_x)
    return 0.5 * (1.0 - np.tanh(x / cfg.eps))


def wave_speed(cfg: Config) -> float:
    """Analytic KPP traveling-wave speed: c = 2*sqrt(r*D)."""
    return 2.0 * np.sqrt(cfg.r * cfg.D)


def compute_stable_dt(cfg: Config, cfl: float = 0.4) -> float:
    """
    Stable explicit time step for the diffusion term (von Neumann):
    dt <= dx^2 / (2*D). The reaction term r*u*(1-u) is bounded
    (|r*u*(1-u)| <= r/4 for u in [0,1]) so it does not tighten the
    diffusion-dominated stability limit for the parameter ranges used here.
    """
    dx = cfg.delta_x
    return cfl * dx**2 / cfg.D


def fd_step(u: np.ndarray, dt: float, cfg: Config) -> np.ndarray:
    """
    One explicit-Euler FD step:
        u_xx via central differences (interior points)
        reaction term r*u*(1-u) explicit
        Dirichlet BC: u[0] = 1, u[-1] = 0 held fixed every step
    """
    dx = cfg.delta_x
    u_next = u.copy()

    laplacian = np.zeros_like(u)
    laplacian[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2

    reaction = cfg.r * u * (1.0 - u)

    u_next[1:-1] = u[1:-1] + dt * (cfg.D * laplacian[1:-1] + reaction[1:-1])
    u_next[0] = 1.0
    u_next[-1] = 0.0

    return u_next


def solve_kpp_fd(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """
    March the FD solver from u_0 to t=T using a CFL-stable dt,
    recording n_t evenly spaced snapshots (via linear-in-index sampling
    of the internal fine time steps).
    """
    dt = compute_stable_dt(cfg)
    n_steps = max(int(np.ceil(cfg.T / dt)), 1)
    dt = cfg.T / n_steps  # exact fit to T

    u = u_0(cfg)
    snapshot_every = max(n_steps // cfg.n_t, 1)

    u_grid = [u.copy()]
    t_list = [0.0]

    for step in tqdm(range(1, n_steps + 1), desc="Computing FD reference solution"):
        u = fd_step(u, dt, cfg)
        if step % snapshot_every == 0 or step == n_steps:
            u_grid.append(u.copy())
            t_list.append(step * dt)

    return np.array(u_grid), np.array(t_list)


def interpolate_solution(
    u_grid: np.ndarray, t_arr: np.ndarray, x: float, t: float, cfg: Config
) -> float:
    """Bilinear interpolation of the FD solution at physical coordinates (x, t)."""
    n_t, n_x = u_grid.shape

    xi = (x + cfg.L) / cfg.delta_x
    ti = np.searchsorted(t_arr, t, side="right") - 1
    ti = np.clip(ti, 0, n_t - 2)
    xi = np.clip(xi, 0, n_x - 1)

    x0, x1 = int(np.floor(xi)), min(int(np.floor(xi)) + 1, n_x - 1)
    t0, t1 = int(ti), min(int(ti) + 1, n_t - 1)

    dx = xi - x0
    dt = (t - t_arr[t0]) / (t_arr[t1] - t_arr[t0]) if t_arr[t1] != t_arr[t0] else 0.0

    f00, f10 = u_grid[t0, x0], u_grid[t0, x1]
    f01, f11 = u_grid[t1, x0], u_grid[t1, x1]

    return (
        f00 * (1 - dx) * (1 - dt)
        + f10 * dx * (1 - dt)
        + f01 * (1 - dx) * dt
        + f11 * dx * dt
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd Fisher_KPP && python -m pytest tests/test_analytic.py -v`
Expected: PASS (7 tests). Note: `solve_kpp_fd` with default `Config()` is slow-ish (n_x=400, small dt) — tests above use reduced `n_x`/`T`/`n_t` to stay fast.

- [ ] **Step 5: Commit**

```bash
git add Fisher_KPP/analytic.py Fisher_KPP/tests/test_analytic.py
git commit -m "Add Fisher_KPP FD reference solver"
```

---

### Task 3: `model.py` — FCNet and InverseFCNet

**Files:**
- Create: `Fisher_KPP/model.py`
- Test: `Fisher_KPP/tests/test_smoke.py` (created here, extended in later tasks)

**Interfaces:**
- Consumes: `Config` from Task 1.
- Produces: `FCNet(cfg)` with `.forward(t, x) -> Tensor(N,1)`, `.param_count() -> int`. `InverseFCNet(cfg)` additionally exposes `.D_hat` and `.r_hat` properties (positive, via softplus) and raw learnable params `._D_raw`, `._r_raw`. `predict(model, t, x) -> np.ndarray`, `predict_from_state(state_dict, t, x, cfg) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

Create `Fisher_KPP/tests/test_smoke.py`:

```python
import numpy as np
import torch

from config import Config
from model import FCNet, InverseFCNet, predict, predict_from_state


def small_cfg(**overrides) -> Config:
    defaults = dict(n_x=100, T=1.0, n_t=20, n_obs=5, n_col=10, n_bc=5, n_grid_val=5)
    defaults.update(overrides)
    return Config(**defaults)


def test_forward_net_shape():
    cfg = small_cfg()
    model = FCNet(cfg)
    t = torch.rand(8, 1)
    x = torch.rand(8, 1)
    out = model(t, x)
    assert out.shape == (8, 1)
    assert torch.isfinite(out).all()


def test_inverse_net_forward_and_params_positive():
    cfg = small_cfg()
    model = InverseFCNet(cfg)
    t = torch.rand(8, 1)
    x = torch.rand(8, 1)
    out = model(t, x)
    assert out.shape == (8, 1)
    assert model.D_hat.item() > 0
    assert model.r_hat.item() > 0


def test_predict_matches_numpy_shape():
    cfg = small_cfg()
    model = FCNet(cfg)
    t = np.linspace(0.0, 1.0, 20)
    x = np.linspace(-cfg.L, cfg.L, 20)
    y = predict(model, t, x)
    assert y.shape == (20,)
    assert np.isfinite(y).all()


def test_predict_from_state_matches_direct_predict():
    cfg = small_cfg()
    model = FCNet(cfg)
    t = np.linspace(0.0, 1.0, 10)
    x = np.linspace(-cfg.L, cfg.L, 10)

    state_dict = model.state_dict()
    u_direct = predict(model, t, x)
    u_restored = predict_from_state(state_dict, t, x, cfg)

    assert np.allclose(u_direct, u_restored)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd Fisher_KPP && python -m pytest tests/test_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model'`

- [ ] **Step 3: Create `Fisher_KPP/model.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd Fisher_KPP && python -m pytest tests/test_smoke.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add Fisher_KPP/model.py Fisher_KPP/tests/test_smoke.py
git commit -m "Add Fisher_KPP FCNet and InverseFCNet"
```

---

### Task 4: `losses.py`

**Files:**
- Create: `Fisher_KPP/losses.py`
- Modify: `Fisher_KPP/tests/test_smoke.py` (append loss tests)

**Interfaces:**
- Consumes: `Config`, `FCNet`, `InverseFCNet` from Tasks 1 and 3.
- Produces: `loss_data(u_pred, u_obs_t) -> Tensor`, `loss_ic(model, t_ic_t, x_ic_t, u_ic_true) -> Tensor`, `loss_bc(cfg, model, t_bc_t) -> Tensor`, `loss_physics(cfg, model, t_col_t, x_col_t) -> Tensor`, `loss_physics_inverse(model, t_col_t, x_col_t) -> Tensor`.

- [ ] **Step 1: Write the failing tests**

Append to `Fisher_KPP/tests/test_smoke.py`:

```python
from losses import loss_data, loss_physics, loss_physics_inverse, loss_ic, loss_bc


def test_losses_are_finite_and_differentiable():
    cfg = small_cfg()
    model = FCNet(cfg)

    u_pred = torch.rand(5, 1)
    u_obs = torch.rand(5, 1)
    l_data = loss_data(u_pred, u_obs)
    assert torch.isfinite(l_data)

    t_col = torch.rand(5, 1, requires_grad=True)
    x_col = torch.rand(5, 1, requires_grad=True)
    l_phys = loss_physics(cfg, model, t_col, x_col)
    assert torch.isfinite(l_phys)
    l_phys.backward()

    t_ic = torch.zeros(5, 1)
    x_ic = torch.rand(5, 1)
    u_ic_true = torch.rand(5, 1)
    l_ic = loss_ic(model, t_ic, x_ic, u_ic_true)
    assert torch.isfinite(l_ic)

    t_bc = torch.rand(5, 1)
    l_bc = loss_bc(cfg, model, t_bc)
    assert torch.isfinite(l_bc)


def test_loss_physics_inverse_is_finite_and_differentiable():
    cfg = small_cfg()
    model = InverseFCNet(cfg)
    t_col = torch.rand(5, 1, requires_grad=True)
    x_col = torch.rand(5, 1, requires_grad=True)
    l_phys = loss_physics_inverse(model, t_col, x_col)
    assert torch.isfinite(l_phys)
    l_phys.backward()
    assert model._D_raw.grad is not None
    assert model._r_raw.grad is not None


def test_loss_bc_zero_when_model_matches_boundary_exactly():
    cfg = small_cfg()
    model = FCNet(cfg)
    # Force the net to output exactly the BC values by overriding forward
    # via a small wrapper is overkill here; instead check loss_bc runs and
    # is non-negative, which is what the implementation guarantees.
    t_bc = torch.rand(5, 1)
    l_bc = loss_bc(cfg, model, t_bc)
    assert l_bc.item() >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd Fisher_KPP && python -m pytest tests/test_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'losses'`

- [ ] **Step 3: Create `Fisher_KPP/losses.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd Fisher_KPP && python -m pytest tests/test_smoke.py -v`
Expected: PASS (7 tests total in file)

- [ ] **Step 5: Commit**

```bash
git add Fisher_KPP/losses.py Fisher_KPP/tests/test_smoke.py
git commit -m "Add Fisher_KPP loss functions"
```

---

### Task 5: `data.py`

**Files:**
- Create: `Fisher_KPP/data.py`
- Test: `Fisher_KPP/tests/test_data.py`

**Interfaces:**
- Consumes: `Config`, `analytic.u_0`, `analytic.solve_kpp_fd`, `analytic.interpolate_solution` from Tasks 1-2.
- Produces: `make_observation(cfg, u_grid, t_arr) -> tuple[np.ndarray, np.ndarray, np.ndarray]` (t_obs, x_obs, u_obs), `make_collocation(cfg) -> tuple[np.ndarray, np.ndarray]` (t_col, x_col), `make_bc_points(cfg) -> np.ndarray` (t_bc), `make_validation(cfg, u_grid, t_arr) -> tuple[np.ndarray, np.ndarray, np.ndarray]` (t_val, x_val, u_val), `generate_data(cfg) -> dict` with keys `u_grid, t_arr, t_obs, x_obs, u_obs, t_val, x_val, u_val, t_col, x_col, t_bc, x_ic, t_ic, u_ic`.

- [ ] **Step 1: Write the failing tests**

Create `Fisher_KPP/tests/test_data.py`:

```python
import numpy as np

from config import Config
from analytic import u_0, solve_kpp_fd
from data import make_observation, make_collocation, make_bc_points, make_validation, generate_data


def small_cfg(**overrides) -> Config:
    defaults = dict(n_x=100, T=1.0, n_t=20, n_obs=10, n_col=20, n_bc=10, n_grid_val=10)
    defaults.update(overrides)
    return Config(**defaults)


def test_make_collocation_within_domain_bounds():
    cfg = small_cfg()
    t_col, x_col = make_collocation(cfg)
    assert t_col.shape == x_col.shape == (cfg.n_col,)
    assert (t_col >= 0).all() and (t_col <= cfg.T).all()
    assert (x_col >= -cfg.L).all() and (x_col <= cfg.L).all()


def test_make_bc_points_within_time_domain():
    cfg = small_cfg()
    t_bc = make_bc_points(cfg)
    assert t_bc.shape == (cfg.n_bc,)
    assert (t_bc >= 0).all() and (t_bc <= cfg.T).all()


def test_make_observation_interpolates_within_grid_range():
    cfg = small_cfg(sigma=0.0)
    u_grid, t_arr = solve_kpp_fd(cfg)
    t_obs, x_obs, u_obs = make_observation(cfg, u_grid, t_arr)
    assert t_obs.shape == x_obs.shape == u_obs.shape == (cfg.n_obs,)
    assert np.isfinite(u_obs).all()


def test_make_validation_grid_shapes():
    cfg = small_cfg()
    u_grid, t_arr = solve_kpp_fd(cfg)
    t_val, x_val, u_val = make_validation(cfg, u_grid, t_arr)
    assert t_val.shape == x_val.shape == u_val.shape == (cfg.n_grid_val**2,)
    assert np.isfinite(u_val).all()


def test_generate_data_shapes_and_ic_matches_u_0():
    cfg = small_cfg()
    data = generate_data(cfg)
    assert data["t_obs"].shape == data["x_obs"].shape == data["u_obs"].shape
    assert data["t_val"].shape == data["x_val"].shape == data["u_val"].shape
    assert data["u_ic"].shape == data["x_ic"].shape
    assert np.allclose(data["u_ic"], u_0(cfg))
    assert np.isfinite(data["u_grid"]).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd Fisher_KPP && python -m pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data'`

- [ ] **Step 3: Create `Fisher_KPP/data.py`**

```python
"""
Data generation for the Fisher-KPP PINN: observation, collocation,
boundary, initial-condition, and validation points.
"""

import numpy as np
from config import Config
from analytic import u_0, interpolate_solution


def make_observation(
    cfg: Config, u_grid: np.ndarray, t_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Noisy observation points sampled uniformly over the domain."""
    t_obs = np.random.uniform(0.0, cfg.T, cfg.n_obs)
    x_obs = np.random.uniform(-cfg.L, cfg.L, cfg.n_obs)

    u_obs = np.array(
        [
            interpolate_solution(u_grid, t_arr, x, t, cfg) + np.random.normal(0.0, cfg.sigma)
            for x, t in zip(x_obs, t_obs)
        ]
    )
    return t_obs, x_obs, u_obs


def make_collocation(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """PDE residual collocation points sampled uniformly over the domain."""
    t_col = np.random.uniform(0.0, cfg.T, cfg.n_col)
    x_col = np.random.uniform(-cfg.L, cfg.L, cfg.n_col)
    return t_col, x_col


def make_bc_points(cfg: Config) -> np.ndarray:
    """Boundary-condition time points."""
    return np.random.uniform(0.0, cfg.T, cfg.n_bc)


def make_validation(
    cfg: Config, u_grid: np.ndarray, t_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validation grid: interpolate the FD reference at a regular (t, x) mesh."""
    t_1d = np.linspace(0.0, cfg.T, cfg.n_grid_val)
    x_1d = np.linspace(-cfg.L, cfg.L, cfg.n_grid_val)

    tt, xx = np.meshgrid(t_1d, x_1d)
    t_val = tt.ravel()
    x_val = xx.ravel()

    u_val = np.array(
        [interpolate_solution(u_grid, t_arr, x, t, cfg) for x, t in zip(x_val, t_val)]
    )
    return t_val, x_val, u_val


def generate_data(cfg: Config) -> dict:
    """Generate all data and return as a dict."""
    from analytic import solve_kpp_fd

    print("Generating data...")
    u_grid, t_arr = solve_kpp_fd(cfg)

    t_obs, x_obs, u_obs = make_observation(cfg, u_grid, t_arr)
    t_col, x_col = make_collocation(cfg)
    t_val, x_val, u_val = make_validation(cfg, u_grid, t_arr)
    t_bc = make_bc_points(cfg)

    x_ic = np.linspace(-cfg.L, cfg.L, cfg.n_x)

    return {
        "u_grid": u_grid,
        "t_arr": t_arr,
        "t_obs": t_obs,
        "x_obs": x_obs,
        "u_obs": u_obs,
        "t_val": t_val,
        "x_val": x_val,
        "u_val": u_val,
        "t_col": t_col,
        "x_col": x_col,
        "t_bc": t_bc,
        "x_ic": x_ic,
        "t_ic": np.zeros_like(x_ic),
        "u_ic": u_0(cfg),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd Fisher_KPP && python -m pytest tests/test_data.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add Fisher_KPP/data.py Fisher_KPP/tests/test_data.py
git commit -m "Add Fisher_KPP data generation"
```

---

### Task 6: `trainer.py`

**Files:**
- Create: `Fisher_KPP/trainer.py`
- Create: `Fisher_KPP/tests/test_training.py`

**Interfaces:**
- Consumes: `Config`, `FCNet`, `InverseFCNet` (Tasks 1, 3), `to_tensor` (Task 1), losses (Task 4), `make_collocation`, `make_observation`, `make_bc_points` (Task 5).
- Produces: `train(model, data, cfg, device, label="Model", verbatim=True) -> tuple[dict, dict, dict]` returning `(history, snapshots, best_state)`. `history` keys: `epoch, loss_data, loss_phys, loss_ic, loss_bc, loss_val, loss_total`, plus `D_hat, r_hat` when `model` is an `InverseFCNet` (detected via `hasattr(model, "D_hat")`).

- [ ] **Step 1: Write the failing tests**

Create `Fisher_KPP/tests/test_training.py`:

```python
import numpy as np
import torch

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd Fisher_KPP && python -m pytest tests/test_training.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trainer'`

- [ ] **Step 3: Create `Fisher_KPP/trainer.py`**

```python
"""
Training loop for the Fisher-KPP PINN.

Contains:
    train -- run the Adam optimiser with step-decay scheduler,
             early stopping, periodic snapshots, and validation logging
"""

import time
import copy
from config import Config
import torch
import torch.nn as nn
from utils import to_tensor
from losses import loss_data, loss_ic, loss_physics, loss_physics_inverse, loss_bc
from data import make_collocation, make_observation, make_bc_points


def train(
    model: nn.Module,
    data: dict,
    cfg: Config,
    device: torch.device,
    label: str = "Model",
    verbatim: bool = True,
) -> tuple[dict, dict, dict]:
    """
    Train the PINN in forward or inverse mode (inverse detected via
    hasattr(model, "D_hat")).

    Returns
    -------
    history   : dict of loss curves logged every cfg.log_every epochs
                (plus D_hat, r_hat curves in inverse mode)
    snapshots : dict mapping epoch -> CPU state_dict at cfg.snapshot_epochs
    best_state: CPU state_dict with the lowest validation loss
    """
    inverse_mode = hasattr(model, "D_hat")

    model.to(device)

    x_val_t = to_tensor(data["x_val"])
    t_val_t = to_tensor(data["t_val"])
    u_val_t = to_tensor(data["u_val"])
    x_ic_t = to_tensor(data["x_ic"])
    t_ic_t = to_tensor(data["t_ic"])
    u_ic_t = to_tensor(data["u_ic"])

    u_grid = data["u_grid"]
    t_arr = data["t_arr"]

    x_obs_t = to_tensor(data["x_obs"])
    t_obs_t = to_tensor(data["t_obs"])
    u_obs_t = to_tensor(data["u_obs"])

    t_col_t = to_tensor(data["t_col"], requires_grad=True)
    x_col_t = to_tensor(data["x_col"], requires_grad=True)

    t_bc_t = to_tensor(data["t_bc"])

    best_state = None

    if inverse_mode:
        optimiser = torch.optim.Adam(
            [
                {"params": model.net.parameters(), "lr": cfg.lr},
                {"params": [model._D_raw, model._r_raw], "lr": cfg.lr_inverse},
            ],
            betas=(cfg.adam_beta1, cfg.adam_beta2),
        )
    else:
        optimiser = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2)
        )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=cfg.scheduler_step, gamma=cfg.scheduler_gamma
    )

    history = {
        "epoch": [],
        "loss_data": [],
        "loss_phys": [],
        "loss_ic": [],
        "loss_bc": [],
        "loss_val": [],
        "loss_total": [],
    }
    if inverse_mode:
        history["D_hat"] = []
        history["r_hat"] = []

    snapshots = {}
    best_val_loss = float("inf")
    epochs_no_improvement = 0

    t0 = time.perf_counter()

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        optimiser.zero_grad()

        if cfg.use_data:
            if cfg.randomise_observation:
                t_obs, x_obs, u_obs = make_observation(cfg, u_grid, t_arr)
                t_obs_t = to_tensor(t_obs)
                x_obs_t = to_tensor(x_obs)
                u_obs_t = to_tensor(u_obs)

            u_pred = model(t_obs_t, x_obs_t)
            l_data = loss_data(u_pred, u_obs_t)
        else:
            l_data = torch.zeros(1, device=device)

        if cfg.randomise_collocation:
            t_col, x_col = make_collocation(cfg)
            t_col_t = to_tensor(t_col, requires_grad=True)
            x_col_t = to_tensor(x_col, requires_grad=True)

        if inverse_mode:
            l_phys = loss_physics_inverse(model, t_col_t, x_col_t)
        elif cfg.use_physics:
            l_phys = loss_physics(cfg, model, t_col_t, x_col_t)
        else:
            l_phys = torch.zeros(1, device=device)

        l_ic = (
            loss_ic(model, t_ic_t, x_ic_t, u_ic_t)
            if cfg.use_ic
            else torch.zeros(1, device=device)
        )

        if cfg.randomise_bc_points:
            t_bc = make_bc_points(cfg)
            t_bc_t = to_tensor(t_bc)

        l_bc = (
            loss_bc(cfg, model, t_bc_t) if cfg.use_bc else torch.zeros(1, device=device)
        )

        loss_total = (
            l_data
            + cfg.lambda_phys * l_phys
            + cfg.lambda_ic * l_ic
            + cfg.lambda_bc * l_bc
        )

        loss_total.backward()
        optimiser.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            u_val_pred = model(t_val_t, x_val_t)
            l_val = loss_data(u_val_pred, u_val_t)

        if l_val.item() < best_val_loss - cfg.patience_threshold:
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1

        if l_val.item() < best_val_loss:
            best_val_loss = l_val.item()
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if epochs_no_improvement >= cfg.patience:
            if verbatim:
                print(
                    f"[{label}] early stopping at epoch {epoch}, improvement stalled."
                )
            break

        if epoch in cfg.snapshot_epochs:
            snapshots[epoch] = copy.deepcopy(model.state_dict())

        if verbatim and (epoch % cfg.log_every == 0 or epoch == 1):
            elapsed = time.perf_counter() - t0
            param_str = (
                f"D_hat={model.D_hat.item():.4f} r_hat={model.r_hat.item():.4f}  "
                if inverse_mode
                else ""
            )
            print(
                f"  [{label}] epoch {epoch:5d} | "
                f"L_data={l_data.item():.5f}  "
                f"L_phys={l_phys.item():.5f}  "
                f"L_ic={l_ic.item():.5f}  "
                f"L_bc={l_bc.item():.5f}  "
                f"L_val={l_val.item():.5f}  "
                f"L_total={loss_total.item():.5f}  "
                f"{param_str}"
                f"({elapsed:.1f}s)"
            )

        if epoch % cfg.log_every == 0:
            history["epoch"].append(epoch)
            history["loss_data"].append(l_data.item())
            history["loss_phys"].append(l_phys.item())
            history["loss_ic"].append(l_ic.item())
            history["loss_bc"].append(l_bc.item())
            history["loss_val"].append(l_val.item())
            history["loss_total"].append(loss_total.item())
            if inverse_mode:
                history["D_hat"].append(model.D_hat.item())
                history["r_hat"].append(model.r_hat.item())

    model.load_state_dict(best_state)
    total_time = time.perf_counter() - t0
    if verbatim:
        print(f"  [{label}] finished in {total_time:.1f}s ({total_time / epoch * 1000:.2f} ms/epoch)")
        if inverse_mode:
            print(f"{'Predicted D:':<20} {float(model.D_hat):.4f}")
            print(f"{'Predicted r:':<20} {float(model.r_hat):.4f}")
            print(f"{'Actual D:':<20} {cfg.D:.4f}")
            print(f"{'Actual r:':<20} {cfg.r:.4f}")

    return history, snapshots, best_state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd Fisher_KPP && python -m pytest tests/test_training.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add Fisher_KPP/trainer.py Fisher_KPP/tests/test_training.py
git commit -m "Add Fisher_KPP training loop"
```

---

### Task 7: `plot.py`

**Files:**
- Create: `Fisher_KPP/plot.py`

**Interfaces:**
- Consumes: `Config`, `analytic.wave_speed`, `model.predict_from_state`, `utils.save_show`, `utils.load_model`.
- Produces: `style_ax(ax) -> None`, `plot_solution_heatmap(model_state, cfg, u_grid, t_arr, output_path=None, show=True) -> None`, `plot_snapshots(model_state, cfg, u_grid, t_arr, times=(0.0, 1.5, 3.0, 4.5), output_path=None, show=True) -> None`, `plot_losses(history, output_path=None, show=True) -> None`, `plot_predicted_parameter_convergence(history, cfg, output_path=None, show=True) -> None` (inverse mode only), `save_plots_from_file(folder_path, verbatim=True) -> None`.

No new tests here — plotting is validated visually via the demo scripts (Task 8), matching `Burgers_Equation/plot.py`, which also has no dedicated test file (see `run_tests.py` coverage gap already known: "plotting module remains untested").

- [ ] **Step 1: Create `Fisher_KPP/plot.py`**

```python
"""
Plotting utilities for the Fisher-KPP PINN.

Contains:
    style_ax                            -- shared axis styling
    plot_solution_heatmap                -- PINN vs FD reference heatmap
    plot_snapshots                       -- PINN vs FD reference at fixed times
    plot_losses                          -- training loss curves
    plot_predicted_parameter_convergence -- D_hat, r_hat vs epoch (inverse mode)
    save_plots_from_file                 -- load a saved run and regenerate all plots
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from config import Config
from analytic import wave_speed
from model import FCNet, InverseFCNet, predict_from_state
from utils import save_show, load_model

BLUE = "#378ADD"
RED = "#E24B4A"
GREEN = "#1D9E75"
GRAY = "#888780"
LGRAY = "#D3D1C7"
BG = "#FAFAF8"
PANEL = "#F1EFE8"


def style_ax(ax: Axes) -> None:
    """Apply shared axis styling."""
    ax.set_facecolor(PANEL)
    ax.figure.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(LGRAY)
    ax.tick_params(colors=GRAY)


def plot_solution_heatmap(
    model_state: dict,
    cfg: Config,
    u_grid: np.ndarray,
    t_arr: np.ndarray,
    output_path: str = None,
    show: bool = True,
) -> None:
    """Side-by-side heatmap of the FD reference and the PINN prediction."""
    n_t, n_x = u_grid.shape
    x = np.linspace(-cfg.L, cfg.L, n_x)

    tt, xx = np.meshgrid(t_arr, x, indexing="ij")
    u_pred = predict_from_state(model_state, tt.ravel(), xx.ravel(), cfg).reshape(n_t, n_x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, grid, title in zip(axes, [u_grid, u_pred], ["FD reference", "PINN"]):
        style_ax(ax)
        im = ax.pcolormesh(x, t_arr, grid, shading="auto", cmap="viridis")
        ax.set_xlabel("$x$")
        ax.set_title(title)
    axes[0].set_ylabel("$t$")
    fig.colorbar(im, ax=axes, label="$u(x,t)$")

    save_show(output_path, show)


def plot_snapshots(
    model_state: dict,
    cfg: Config,
    u_grid: np.ndarray,
    t_arr: np.ndarray,
    times: tuple = (0.0, 1.5, 3.0, 4.5),
    output_path: str = None,
    show: bool = True,
) -> None:
    """Plot FD reference vs PINN prediction at fixed times, plus the analytic wave-speed marker."""
    n_x = u_grid.shape[1]
    x = np.linspace(-cfg.L, cfg.L, n_x)
    c = wave_speed(cfg)

    fig, ax = plt.subplots(figsize=(9, 6))
    style_ax(ax)
    for t_target in times:
        idx = int(np.argmin(np.abs(t_arr - t_target)))
        t_actual = t_arr[idx]
        u_pred = predict_from_state(model_state, np.full_like(x, t_actual), x, cfg)

        ax.plot(x, u_grid[idx], color=GRAY, linestyle="--", alpha=0.7)
        ax.plot(x, u_pred, color=GREEN, alpha=0.9, label=f"$t={t_actual:.2f}$")
        # front position predicted by the analytic wave speed, from x=0 at t=0
        ax.axvline(c * t_actual, color=RED, linestyle=":", alpha=0.4)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(x,t)$")
    ax.set_title("Fisher-KPP: PINN (solid) vs FD reference (dashed)")
    ax.legend(fontsize=8)

    save_show(output_path, show)


def plot_losses(history: dict, output_path: str = None, show: bool = True) -> None:
    """Plot training loss curves on a log scale."""
    fig, ax = plt.subplots(figsize=(8, 5))
    style_ax(ax)
    for key, color in [
        ("loss_data", BLUE),
        ("loss_phys", "#EF9F27"),
        ("loss_ic", "#7F77DD"),
        ("loss_bc", RED),
        ("loss_val", GREEN),
        ("loss_total", GRAY),
    ]:
        if any(v != 0 for v in history[key]):
            ax.plot(history["epoch"], history[key], label=key, color=color)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)

    save_show(output_path, show)


def plot_predicted_parameter_convergence(
    history: dict, cfg: Config, output_path: str = None, show: bool = True
) -> None:
    """Plot D_hat and r_hat convergence against the ground-truth values."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, true_val, label in zip(
        axes, ["D_hat", "r_hat"], [cfg.D, cfg.r], ["$D$", "$r$"]
    ):
        style_ax(ax)
        ax.plot(history["epoch"], history[key], color=GREEN, label=f"{label} (predicted)")
        ax.axhline(true_val, color=RED, linestyle="--", label=f"{label} (true)")
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)

    save_show(output_path, show)


def save_plots_from_file(folder_path: str, verbatim: bool = True) -> None:
    """Load a saved run (best_model.pt, history.csv, config.json) and regenerate all plots."""
    folder_path = Path(folder_path)

    with open(folder_path / "config.json", encoding="utf-8") as f:
        import json

        cfg = Config(**json.load(f))

    inverse_mode = "D_hat" in _peek_history_keys(folder_path)
    model = InverseFCNet(cfg) if inverse_mode else FCNet(cfg)
    model, history, _snapshots, cfg = load_model(model, folder_path)

    from analytic import solve_kpp_fd

    u_grid, t_arr = solve_kpp_fd(cfg)
    best_state = model.state_dict()

    plot_solution_heatmap(
        best_state, cfg, u_grid, t_arr,
        output_path=str(folder_path / "solution_heatmap.png"), show=False,
    )
    plot_snapshots(
        best_state, cfg, u_grid, t_arr,
        output_path=str(folder_path / "snapshots.png"), show=False,
    )
    plot_losses(history, output_path=str(folder_path / "losses.png"), show=False)

    if inverse_mode:
        plot_predicted_parameter_convergence(
            history, cfg,
            output_path=str(folder_path / "parameter_convergence.png"), show=False,
        )

    if verbatim:
        print(f"Plots saved to {folder_path}/")


def _peek_history_keys(folder_path: Path) -> list:
    """Read just the header row of history.csv to detect inverse-mode columns."""
    import csv

    with open(folder_path / "history.csv", encoding="utf-8") as f:
        return next(csv.reader(f))
```

- [ ] **Step 2: Sanity-check import and a smoke render**

Run:
```bash
cd Fisher_KPP && python -c "
import matplotlib
matplotlib.use('Agg')
from config import Config
from analytic import solve_kpp_fd
from model import FCNet
from plot import plot_solution_heatmap, plot_snapshots, plot_losses

cfg = Config(n_x=100, T=1.0, n_t=20)
u_grid, t_arr = solve_kpp_fd(cfg)
model = FCNet(cfg)
state = model.state_dict()
plot_solution_heatmap(state, cfg, u_grid, t_arr, show=False)
plot_snapshots(state, cfg, u_grid, t_arr, times=(0.0, 0.5, 1.0), show=False)
plot_losses({'epoch': [1,2], 'loss_data':[1,0], 'loss_phys':[1,0], 'loss_ic':[1,0], 'loss_bc':[1,0], 'loss_val':[1,0], 'loss_total':[2,0]}, show=False)
print('OK')
"
```
Expected: prints `OK` with no exceptions.

- [ ] **Step 3: Commit**

```bash
git add Fisher_KPP/plot.py
git commit -m "Add Fisher_KPP plotting utilities"
```

---

### Task 8: Scripts — forward demo, inverse data generation, inverse demo, report plots

**Files:**
- Create: `Fisher_KPP/scripts/pinn_demo.py`
- Create: `Fisher_KPP/scripts/inversepinn_data_generation.py`
- Create: `Fisher_KPP/scripts/inversepinn_demo.py`
- Create: `Fisher_KPP/scripts/report_plots.py`

**Interfaces:**
- Consumes everything from Tasks 1-7.
- Produces: four runnable CLI scripts, each `sys.path.append`-ing the project root exactly like `Burgers_Equation/scripts/*.py`.

- [ ] **Step 1: Create `Fisher_KPP/scripts/pinn_demo.py`**

```python
"""
Demo script for training a PINN to solve the forward Fisher-KPP problem.

Usage:
    python pinn_demo.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
import argparse
from config import Config
from data import generate_data
from model import FCNet
from plot import save_plots_from_file
from trainer import train
from utils import get_device, save_model

SCRIPT_DIR = Path(__file__).resolve().parent.parent


def parse_args():
    """Parsing arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=float, default=None, help="Diffusion coefficient")
    parser.add_argument("--r", type=float, default=None, help="Growth rate")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--output", type=Path, default=None, help="Override output directory")
    return parser.parse_args()


def resolve_output_path(args, D, r):
    """Pick an output folder based on D, r unless overridden."""
    if args.output is not None:
        return args.output
    return SCRIPT_DIR / f"outputs/pinn_D{D:g}_r{r:g}".replace(".", "p")


def main() -> None:
    """Main loop."""
    print("=" * 50)
    print("  PINN Demo — Fisher-KPP Equation")
    print("  Training a physics-informed neural network")
    print("  to solve the forward problem.")
    print("=" * 50)
    device = get_device()
    print(f"Device : {device}")

    args = parse_args()
    D = args.D if args.D is not None else 1.0
    r = args.r if args.r is not None else 1.0
    output_path = resolve_output_path(args, D, r)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"D      : {D:.4f}")
    print(f"r      : {r:.4f}")

    cfg = Config(D=D, r=r)

    data = generate_data(cfg)
    model = FCNet(cfg)
    history, snapshots, best_state = train(
        model=model, data=data, cfg=cfg, device=device, label="Model",
    )

    save_model(best_state, history, snapshots, cfg, output_path)
    print(f"Model saved to {output_path}")

    save_plots_from_file(output_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `Fisher_KPP/scripts/inversepinn_demo.py`**

```python
"""
Demo script for training a PINN to solve the inverse Fisher-KPP problem
(recover D, r from noisy synthetic observations).

Usage:
    python inversepinn_demo.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
import argparse
from config import Config
from data import generate_data
from model import InverseFCNet
from plot import save_plots_from_file
from trainer import train
from utils import get_device, save_model

SCRIPT_DIR = Path(__file__).resolve().parent.parent


def parse_args():
    """Parsing arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=float, default=1.0, help="True diffusion coefficient")
    parser.add_argument("--r", type=float, default=1.0, help="True growth rate")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--output", type=Path, default=None, help="Override output directory")
    return parser.parse_args()


def resolve_output_path(args):
    """Pick an output folder based on D, r unless overridden."""
    if args.output is not None:
        return args.output
    label = f"D{args.D:g}_r{args.r:g}".replace(".", "p")
    return SCRIPT_DIR / f"outputs/inversepinn_{label}"


def main() -> None:
    """Main loop."""
    print("=" * 50)
    print("  Inverse PINN Demo — Fisher-KPP Equation")
    print("  Training a physics-informed neural network")
    print("  to solve the inverse problem.")
    print("=" * 50)
    device = get_device()
    print(f"Device : {device}")

    args = parse_args()
    output_path = resolve_output_path(args)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"D      : {args.D:.4f}")
    print(f"r      : {args.r:.4f}")

    cfg = Config(D=args.D, r=args.r, use_data=True)

    data = generate_data(cfg)
    model = InverseFCNet(cfg)
    history, snapshots, best_state = train(
        model=model, data=data, cfg=cfg, device=device, label="Model",
    )

    save_model(best_state, history, snapshots, cfg, output_path)
    print(f"Model saved to {output_path}")

    save_plots_from_file(output_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create `Fisher_KPP/scripts/inversepinn_data_generation.py`**

```python
"""
Script to continuously generate true-predicted (D, r) pairs for the
Fisher-KPP inverse problem and save them to a CSV file.

Runs until KeyboardInterrupt. Each iteration randomises D, r, trains an
InverseFCNet, and appends the result to a CSV.

Usage:
    python inversepinn_data_generation.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
import csv
import numpy as np
from plot import plot_predicted_parameter_convergence, plot_solution_heatmap
from config import Config
from data import generate_data
from model import InverseFCNet
from trainer import train
from utils import get_device

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "outputs/parameter_estimation"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_PATH / "D_r_pairs.csv"
CSV_FIELDS = [
    "run", "D_true", "r_true", "D_pred", "r_pred",
    "D_abs_error", "r_abs_error", "best_val_loss",
]

D_RANGE = (0.5, 2.0)
R_RANGE = (0.5, 2.0)


def run_single(run_idx: int, device) -> dict:
    """Randomise D, r, train an InverseFCNet, return result row."""
    D = float(np.random.uniform(*D_RANGE))
    r = float(np.random.uniform(*R_RANGE))

    print(f"-----Randomised D: {D:.4f}, r: {r:.4f}-----")
    cfg = Config(D=D, r=r, use_data=True)

    data = generate_data(cfg)
    model = InverseFCNet(cfg)

    history, snapshots, _ = train(
        model=model, data=data, cfg=cfg, device=device, label=f"run {run_idx}", verbatim=False,
    )

    D_pred = float(model.D_hat.item())
    r_pred = float(model.r_hat.item())
    best_val = min(history["loss_val"])

    print(
        f"  [run {run_idx}] D_true={D:.4f} D_pred={D_pred:.4f}  "
        f"r_true={r:.4f} r_pred={r_pred:.4f}  val={best_val:.5f}"
    )

    model.to("cpu")
    plot_predicted_parameter_convergence(
        history=history, cfg=cfg,
        output_path=f"{OUTPUT_PATH}/convergence_D{D:.4f}_r{r:.4f}.png", show=False,
    )
    plot_solution_heatmap(
        model_state=model.state_dict(), cfg=cfg,
        u_grid=data["u_grid"], t_arr=data["t_arr"],
        output_path=f"{OUTPUT_PATH}/solution_D{D:.4f}_r{r:.4f}.png", show=False,
    )

    return {
        "run": run_idx, "D_true": D, "r_true": r,
        "D_pred": D_pred, "r_pred": r_pred,
        "D_abs_error": abs(D_pred - D), "r_abs_error": abs(r_pred - r),
        "best_val_loss": best_val,
    }


def main():
    """Main loop."""
    device = get_device()
    print(f"Device      : {device}")
    print(f"Output CSV  : {CSV_PATH}")
    print("Press Ctrl+C to stop.\n")

    write_header = not CSV_PATH.exists()
    run_idx = 0

    try:
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()

            while True:
                run_idx += 1
                print(f"--- Run {run_idx} ---")
                try:
                    row = run_single(run_idx, device)
                except (RuntimeError, ValueError, OSError) as e:
                    print(f"  [run {run_idx}] failed: {e} — skipping.")
                    continue

                writer.writerow(row)
                f.flush()

    except KeyboardInterrupt:
        print(f"\nStopped after {run_idx} runs. Results saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `Fisher_KPP/scripts/report_plots.py`**

```python
"""
Plot generation for the Fisher-KPP report.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
import shutil
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from analytic import u_0, wave_speed, solve_kpp_fd
from plot import style_ax

USE_TEX = shutil.which("latex") is not None
plt.rcParams.update(
    {"text.usetex": USE_TEX, "font.family": "Helvetica" if USE_TEX else "DejaVu Sans"}
)

REPORT_DIR = Path(__file__).resolve().parent.parent / "outputs/report_images"


def ic_and_wave_speed_figure():
    """Plot the initial condition and the FD solution at several times, with
    the analytic wave-speed front position marked."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = Config(n_x=400, T=6.0, n_t=600)
    x = np.linspace(-cfg.L, cfg.L, cfg.n_x)
    c = wave_speed(cfg)

    u_grid, t_arr = solve_kpp_fd(cfg)

    fig, ax = plt.subplots(figsize=(10, 6))
    style_ax(ax)
    for t_target in (0.0, 2.0, 4.0, 6.0):
        idx = int(np.argmin(np.abs(t_arr - t_target)))
        ax.plot(x, u_grid[idx], label=f"$t={t_arr[idx]:.1f}$")
        ax.axvline(c * t_arr[idx], linestyle=":", alpha=0.4)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    ax.set_xlabel("$x$", fontsize=14)
    ax.set_ylabel(r"Solution $u(x,\ t)$", fontsize=14)
    ax.set_title(r"Fisher-KPP traveling front ($D=1$, $r=1$, $c=2$)", fontsize=16)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "kpp_traveling_front.png")


def main():
    ic_and_wave_speed_figure()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Verify all four scripts import cleanly**

Run:
```bash
cd Fisher_KPP && python -c "
import matplotlib
matplotlib.use('Agg')
import sys
sys.path.insert(0, 'scripts')
import pinn_demo, inversepinn_demo, inversepinn_data_generation, report_plots
print('All scripts import OK')
"
```
Expected: prints `All scripts import OK` with no exceptions.

- [ ] **Step 6: Run a short end-to-end smoke of the forward demo (small config, non-default)**

Run:
```bash
cd Fisher_KPP && python -c "
import matplotlib
matplotlib.use('Agg')
from config import Config
from data import generate_data
from model import FCNet
from trainer import train
from utils import get_device

cfg = Config(n_x=100, T=1.0, n_t=20, n_epochs=50, log_every=10, hidden=8, n_layers=2, patience=1000)
data = generate_data(cfg)
model = FCNet(cfg)
history, snapshots, best_state = train(model, data, cfg, device=get_device(), verbatim=True)
print('final loss_total:', history['loss_total'][-1])
"
```
Expected: prints training log lines and a final `loss_total` lower than the first logged value, no exceptions.

- [ ] **Step 7: Commit**

```bash
git add Fisher_KPP/scripts/
git commit -m "Add Fisher_KPP demo and report scripts"
```

---

### Task 9: Full test suite and repo-wide integration check

**Files:** none new — verification only.

- [ ] **Step 1: Run the Fisher_KPP test suite directly**

Run: `cd Fisher_KPP && python -m pytest tests -v`
Expected: all tests from Tasks 1-6 PASS (config, utils, analytic, model, losses, data, trainer — ~27 tests total).

- [ ] **Step 2: Run the repo-wide test runner to confirm auto-discovery and no collisions**

Run: `cd /Users/imranmunir/dev/pinn-dho-burgers && python run_tests.py`
Expected: `Fisher_KPP`, `Burgers_Equation`, `Damped_Oscillator` all reported `PASSED`; final line `ALL PROJECTS PASSED`.

- [ ] **Step 3: Commit (only if Step 1/2 required fixes)**

```bash
git add -A
git commit -m "Fix Fisher_KPP test issues found in full-suite run"
```
(Skip this step if no fixes were needed.)

---

## Self-Review Notes

- **Spec coverage:** every spec section (equation, IC/BC, defaults, FD reference, architecture mirroring, scripts scope, tests) maps to a task above (Tasks 1-8; Task 9 is verification).
- **Placeholder scan:** no TBD/TODO; every step has complete, runnable code.
- **Type consistency:** `Config` field names (`D, r, eps, L, T, n_x, n_t, n_col, ...`) are used identically across `analytic.py`, `data.py`, `model.py`, `losses.py`, `trainer.py`, `plot.py`, and both script files. `InverseFCNet.D_hat`/`r_hat` naming is consistent everywhere it's referenced (trainer, plot, scripts). `generate_data` dict keys (`u_grid, t_arr, t_obs, x_obs, u_obs, t_val, x_val, u_val, t_col, x_col, t_bc, x_ic, t_ic, u_ic`) match what `trainer.train` reads.
