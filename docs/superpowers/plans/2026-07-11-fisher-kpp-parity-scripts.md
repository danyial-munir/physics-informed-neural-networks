# Fisher_KPP Parity Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Optuna hyperparameter tuning, a no-physics ablation, a (D, r) case sweep, and a blind-extrapolation experiment to `Fisher_KPP/`, bringing its script coverage closer to `Burgers_Equation/`'s.

**Architecture:** New scripts under `Fisher_KPP/scripts/`, following the exact structural pattern of their `Burgers_Equation/scripts/` counterparts. One small core addition: a `t_train`/`train_extrap` domain split in `Config` and `data.make_collocation`, needed only for the blind-extrapolation script — mirrors Burgers' `t_dom`/`t_extrap` split.

**Tech Stack:** PyTorch, NumPy, Matplotlib, Optuna, pytest — all already in `pyproject.toml`, no new dependencies.

## Global Constraints

- No changes to `Damped_Oscillator/` or `Burgers_Equation/`.
- No changes to already-implemented core Fisher_KPP modules beyond the `t_train`/`train_extrap` addition in Task 1.
- Fisher_KPP has no IC-regime axis like Burgers' Gauss/N_wave/Step — Optuna tuning is a single study (fixed `D=1.0, r=1.0`), not a loop over regimes.
- `t_train` defaults to `T` when unset (existing forward/inverse behavior unchanged); `train_extrap` defaults to `True` (existing behavior unchanged).
- No new tests for the scripts themselves (matches `Burgers_Equation/scripts/*.py` convention — none of those have dedicated tests either); each script is smoke-verified by running a short/reduced-epoch invocation once during implementation, not committed as an automated test.
- `Fisher_KPP/utils.py` currently has no `load_best_cfg` — Task 1 adds it (needed by `optuna_tuning_plots.py`), mirroring `Burgers_Equation/utils.py`'s version but without the `regime` remapping (Fisher_KPP has no `ic` field).
- Working directory for all Python/pytest commands: use `uv run python ...` / `uv run pytest ...` from the repo root, or `cd Fisher_KPP && uv run pytest tests -v`.

---

## File Structure

```
Fisher_KPP/
├── config.py                              # MODIFY: add t_train, train_extrap
├── data.py                                 # MODIFY: make_collocation domain split
├── utils.py                                 # MODIFY: add load_best_cfg
├── tests/
│   ├── test_config.py                       # MODIFY: t_train default/override tests
│   └── test_data.py                          # MODIFY: make_collocation split tests
└── scripts/
    ├── optuna_tuning.py                       # NEW
    ├── optuna_tuning_plots.py                  # NEW
    ├── pinn_nophys.py                           # NEW
    ├── pinn_cases.py                             # NEW
    └── pinn_blind_extrap.py                       # NEW
```

---

### Task 1: Core domain-split addition (`config.py`, `data.py`, `utils.py`)

**Files:**
- Modify: `Fisher_KPP/config.py`
- Modify: `Fisher_KPP/data.py`
- Modify: `Fisher_KPP/utils.py`
- Modify: `Fisher_KPP/tests/test_config.py`
- Modify: `Fisher_KPP/tests/test_data.py`

**Interfaces:**
- Consumes: existing `Config`, `make_collocation`.
- Produces: `Config.t_train: float` (post-init default `T` if unset), `Config.train_extrap: bool = True`. `make_collocation(cfg)` respects `train_extrap`/`t_train`. `load_best_cfg(json_path: str) -> Config` in `utils.py`.

- [ ] **Step 1: Write the failing tests for `Config.t_train`**

Append to `Fisher_KPP/tests/test_config.py`:

```python
def test_t_train_defaults_to_T_when_unset():
    cfg = Config(T=6.0)
    assert cfg.t_train == 6.0


def test_t_train_preserves_explicit_value():
    cfg = Config(T=6.0, t_train=3.0)
    assert cfg.t_train == 3.0


def test_train_extrap_defaults_true():
    cfg = Config()
    assert cfg.train_extrap is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Fisher_KPP && uv run pytest tests/test_config.py -v`
Expected: FAIL — `TypeError: Config.__init__() got an unexpected keyword argument 't_train'`

- [ ] **Step 3: Add `t_train`/`train_extrap` to `Config`**

In `Fisher_KPP/config.py`, add after the `n_t` field (still inside the Time domain block, before the `delta_x`/`delta_t` properties stay where they are — add the new fields and `__post_init__` at the end of the class):

```python
    # Time domain: t in [0, T]
    T: float = 6.0
    n_t: int = 600  # number of FD reference time steps
    t_train: float = None  # end of the training window; defaults to T
    train_extrap: bool = True  # if False, collocation points are restricted to [0, t_train]
```

At the end of the `Config` class (after `log_every: int = 50`), add:

```python

    def __post_init__(self):
        if self.t_train is None:
            self.t_train = self.T
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Fisher_KPP && uv run pytest tests/test_config.py -v`
Expected: PASS (all tests, including the 3 new ones)

- [ ] **Step 5: Write the failing tests for `make_collocation`'s domain split**

Append to `Fisher_KPP/tests/test_data.py`:

```python
def test_make_collocation_respects_t_train_when_train_extrap_false():
    cfg = small_cfg(t_train=0.5, train_extrap=False)
    t_col, x_col = make_collocation(cfg)
    assert (t_col <= cfg.t_train).all()


def test_make_collocation_ignores_t_train_when_train_extrap_true():
    cfg = small_cfg(t_train=0.1, train_extrap=True)
    t_col, x_col = make_collocation(cfg)
    assert t_col.max() > cfg.t_train  # can exceed t_train since default T=1.0
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd Fisher_KPP && uv run pytest tests/test_data.py -v`
Expected: FAIL — `test_make_collocation_respects_t_train_when_train_extrap_false` fails because `make_collocation` doesn't yet look at `train_extrap`/`t_train` (samples up to `cfg.T` regardless).

- [ ] **Step 7: Update `make_collocation` in `Fisher_KPP/data.py`**

Replace:

```python
def make_collocation(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """PDE residual collocation points sampled uniformly over the domain."""
    t_col = np.random.uniform(0.0, cfg.T, cfg.n_col)
    x_col = np.random.uniform(-cfg.L, cfg.L, cfg.n_col)
    return t_col, x_col
```

with:

```python
def make_collocation(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """
    PDE residual collocation points sampled uniformly over the domain.

    When cfg.train_extrap is False, points are restricted to [0, t_train] --
    used by the blind-extrapolation experiment to test whether the PINN
    generalises the traveling front beyond its training window.
    """
    t_max = cfg.T if cfg.train_extrap else cfg.t_train
    t_col = np.random.uniform(0.0, t_max, cfg.n_col)
    x_col = np.random.uniform(-cfg.L, cfg.L, cfg.n_col)
    return t_col, x_col
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd Fisher_KPP && uv run pytest tests/test_data.py -v`
Expected: PASS (all tests, including the 2 new ones)

- [ ] **Step 9: Add `load_best_cfg` to `Fisher_KPP/utils.py`**

Add after the `save_show` function (before `save_model`):

```python
def load_best_cfg(json_path: str) -> Config:
    """Load best Optuna parameters from a JSON file as a Config object."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return Config(**data["best_params"])
```

- [ ] **Step 10: Write a focused test for `load_best_cfg`**

Append to `Fisher_KPP/tests/test_config.py` (add `import json` and `from pathlib import Path` at the top if not already present, plus `from utils import load_best_cfg`):

```python
def test_load_best_cfg_reads_params_from_json(tmp_path):
    payload = {"best_val": 0.01, "best_params": {"D": 0.5, "r": 2.0, "hidden": 32}}
    json_path = tmp_path / "best_params.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = load_best_cfg(str(json_path))
    assert cfg.D == 0.5
    assert cfg.r == 2.0
    assert cfg.hidden == 32
```

- [ ] **Step 11: Run the full Fisher_KPP suite**

Run: `cd Fisher_KPP && uv run pytest tests -v`
Expected: PASS (all tests, no regressions)

- [ ] **Step 12: Commit**

```bash
git add Fisher_KPP/config.py Fisher_KPP/data.py Fisher_KPP/utils.py Fisher_KPP/tests/test_config.py Fisher_KPP/tests/test_data.py
git commit -m "Add train/extrapolation domain split and load_best_cfg to Fisher_KPP"
```

---

### Task 2: `scripts/pinn_nophys.py` — no-physics ablation

**Files:**
- Create: `Fisher_KPP/scripts/pinn_nophys.py`

**Interfaces:**
- Consumes: `Config`, `data.generate_data`, `model.FCNet`, `trainer.train`, `plot.save_plots_from_file`, `utils.get_device`, `utils.save_model` (all from Task 1's unchanged interfaces plus existing modules).

- [ ] **Step 1: Create `Fisher_KPP/scripts/pinn_nophys.py`**

```python
"""
Solving the Fisher-KPP equation without any physics loss terms
(traditional dense network fit to noisy data only).

Usage:
    python pinn_nophys.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
from config import Config
from data import generate_data
from model import FCNet
from trainer import train
from plot import save_plots_from_file
from utils import get_device, save_model

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "outputs"
OUTPUT_PATH.mkdir(exist_ok=True)


def main() -> None:
    """Main loop."""
    device = get_device()
    print(f"Device : {device}")

    cfg = Config(
        D=1.0,
        r=1.0,
        use_data=True,
        use_physics=False,
        use_ic=False,
        use_bc=False,
    )
    print(cfg)

    output_path = OUTPUT_PATH / "D1_r1_ML"

    data = generate_data(cfg)
    model = FCNet(cfg)
    history, snapshots, best_state = train(
        model=model,
        data=data,
        cfg=cfg,
        device=device,
        label="Model",
    )

    save_model(best_state, history, snapshots, cfg, output_path)
    print(f"Model saved to {output_path}")

    save_plots_from_file(output_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-verify with a reduced-epoch run**

Run:
```bash
cd Fisher_KPP && uv run python -c "
import matplotlib
matplotlib.use('Agg')
from config import Config
from data import generate_data
from model import FCNet
from trainer import train
from utils import get_device

cfg = Config(D=1.0, r=1.0, use_data=True, use_physics=False, use_ic=False, use_bc=False,
             n_x=100, T=1.0, n_t=20, n_epochs=50, log_every=10, hidden=8, n_layers=2, patience=1000)
data = generate_data(cfg)
model = FCNet(cfg)
history, snapshots, best_state = train(model, data, cfg, device=get_device(), verbatim=True)
assert all(lp == 0.0 for lp in history['loss_phys'])
assert all(li == 0.0 for li in history['loss_ic'])
assert all(lb == 0.0 for lb in history['loss_bc'])
print('OK: physics/ic/bc losses are zero, data-only training ran')
"
```
Expected: prints training log lines, then `OK: physics/ic/bc losses are zero, data-only training ran`, no exceptions.

- [ ] **Step 3: Commit**

```bash
git add Fisher_KPP/scripts/pinn_nophys.py
git commit -m "Add Fisher_KPP no-physics ablation script"
```

---

### Task 3: `scripts/pinn_cases.py` — D/r case sweep

**Files:**
- Create: `Fisher_KPP/scripts/pinn_cases.py`

**Interfaces:**
- Consumes: same as Task 2.

- [ ] **Step 1: Create `Fisher_KPP/scripts/pinn_cases.py`**

```python
"""
Solving the Fisher-KPP equation for a sweep of (D, r) combinations.

Usage:
    python pinn_cases.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
from config import Config
from data import generate_data
from model import FCNet
from trainer import train
from plot import save_plots_from_file
from utils import get_device, save_model

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "outputs"
OUTPUT_PATH.mkdir(exist_ok=True)


def train_model(cfg: Config, output_path: Path, device) -> None:
    """Train and save one (D, r) case."""
    print(cfg)

    data = generate_data(cfg)
    model = FCNet(cfg)
    history, snapshots, best_state = train(
        model=model,
        data=data,
        cfg=cfg,
        device=device,
        label="Model",
    )

    save_model(best_state, history, snapshots, cfg, output_path)
    print(f"Model saved to {output_path}")

    save_plots_from_file(output_path)


def main() -> None:
    """Main loop."""
    device = get_device()
    print(f"Device : {device}")

    D_values = [0.5, 1.0, 2.0]
    r_values = [0.5, 1.0, 2.0]

    for D in D_values:
        for r in r_values:
            cfg = Config(D=D, r=r)
            label = f"D{D:g}_r{r:g}".replace(".", "p")
            output_path = OUTPUT_PATH / label
            train_model(cfg, output_path, device)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-verify with a single reduced-epoch case**

Run:
```bash
cd Fisher_KPP && uv run python -c "
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from config import Config
from utils import get_device
import sys
sys.path.insert(0, 'scripts')
from pinn_cases import train_model

cfg = Config(D=0.5, r=2.0, n_x=100, T=1.0, n_t=20, n_epochs=50, log_every=10, hidden=8, n_layers=2, patience=1000)
train_model(cfg, Path('/tmp/fisher_kpp_case_smoketest'), get_device())
print('OK')
"
```
Expected: prints training log lines, then `OK`, no exceptions.

- [ ] **Step 3: Commit**

```bash
git add Fisher_KPP/scripts/pinn_cases.py
git commit -m "Add Fisher_KPP D/r case sweep script"
```

---

### Task 4: `scripts/pinn_blind_extrap.py` — blind extrapolation

**Files:**
- Create: `Fisher_KPP/scripts/pinn_blind_extrap.py`

**Interfaces:**
- Consumes: `Config.t_train`, `Config.train_extrap` from Task 1.

- [ ] **Step 1: Create `Fisher_KPP/scripts/pinn_blind_extrap.py`**

```python
"""
Solving the Fisher-KPP equation without collocation points in the
extrapolated (untrained) time region, to test whether the PINN
generalises the traveling front beyond its training window.

Usage:
    python pinn_blind_extrap.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
from config import Config
from data import generate_data
from model import FCNet
from trainer import train
from plot import save_plots_from_file
from utils import get_device, save_model

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "outputs"
OUTPUT_PATH.mkdir(exist_ok=True)


def main() -> None:
    """Main loop."""
    device = get_device()
    print(f"Device : {device}")

    cfg = Config(D=1.0, r=1.0, t_train=3.0, train_extrap=False)
    print(cfg)

    output_path = OUTPUT_PATH / "D1_r1_extrapblind"

    data = generate_data(cfg)
    model = FCNet(cfg)
    history, snapshots, best_state = train(
        model=model,
        data=data,
        cfg=cfg,
        device=device,
        label="Model",
    )

    save_model(best_state, history, snapshots, cfg, output_path)
    print(f"Model saved to {output_path}")

    save_plots_from_file(output_path)


if __name__ == "__main__":
    main()
```

Note: `t_train=3.0` is half of the default `T=6.0` — collocation points only cover the first half of the time domain; IC/BC/validation still span the full `[0, 6.0]`, so `save_plots_from_file`'s `snapshots.png` shows the front's predicted position vs. FD reference across the full domain, making the extrapolation error visible past `t=3.0`.

- [ ] **Step 2: Smoke-verify with a reduced-epoch run**

Run:
```bash
cd Fisher_KPP && uv run python -c "
import matplotlib
matplotlib.use('Agg')
from config import Config
from data import generate_data
from model import FCNet
from trainer import train
from utils import get_device

cfg = Config(D=1.0, r=1.0, t_train=0.5, train_extrap=False,
             n_x=100, T=1.0, n_t=20, n_epochs=50, log_every=10, hidden=8, n_layers=2, patience=1000)
data = generate_data(cfg)
model = FCNet(cfg)
history, snapshots, best_state = train(model, data, cfg, device=get_device(), verbatim=True)
print('OK: trained with collocation restricted to t_train=0.5 out of T=1.0')
"
```
Expected: prints training log lines, then `OK: trained with collocation restricted to t_train=0.5 out of T=1.0`, no exceptions.

- [ ] **Step 3: Commit**

```bash
git add Fisher_KPP/scripts/pinn_blind_extrap.py
git commit -m "Add Fisher_KPP blind extrapolation script"
```

---

### Task 5: Optuna tuning + plots

**Files:**
- Create: `Fisher_KPP/scripts/optuna_tuning.py`
- Create: `Fisher_KPP/scripts/optuna_tuning_plots.py`

**Interfaces:**
- Consumes: `Config`, `model.FCNet`, `trainer.train`, `data.generate_data`, `utils.get_device`, `utils.load_best_cfg` (Task 1), `utils.save_model`, `plot.save_plots_from_file`.

- [ ] **Step 1: Create `Fisher_KPP/scripts/optuna_tuning.py`**

```python
"""
Optuna TPE hyperparameter search for the Fisher-KPP equation.

Usage:
    python optuna_tuning.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
import random
import csv
import json
import numpy as np
import torch
import optuna

from config import Config
from model import FCNet
from trainer import train
from data import generate_data
from utils import get_device

# -- settings --------------------------------------------------------------
SEED = 42
N_TRIALS = 500

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "outputs/optuna_tuning"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_PATH / "optuna_results.csv"
CSV_FIELDS = [
    "trial",
    "state",
    "rmse",
    "hidden",
    "n_layers",
    "lambda_phys",
    "lambda_ic",
    "lambda_bc",
    "lr",
    "scheduler_gamma",
    "scheduler_step",
]


def set_seed(seed: int) -> None:
    """Set seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


_base_cfg = Config()
_base_data = generate_data(_base_cfg)


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: train and return best validation loss."""
    set_seed(SEED)

    cfg = Config(
        D=_base_cfg.D,
        r=_base_cfg.r,
        hidden=trial.suggest_int("hidden", 16, 128, step=16),
        n_layers=trial.suggest_int("n_layers", 2, 10),
        lambda_phys=trial.suggest_float("lambda_phys", 1e-1, 1e2, log=True),
        lambda_ic=trial.suggest_float("lambda_ic", 1e1, 1e3, log=True),
        lambda_bc=trial.suggest_float("lambda_bc", 1e1, 1e3, log=True),
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        scheduler_gamma=trial.suggest_float("scheduler_gamma", 0.3, 0.9),
        scheduler_step=trial.suggest_int("scheduler_step", 1000, 5000, step=1000),
        use_data=True,
    )

    device = get_device()
    model = FCNet(cfg)

    history, _, _ = train(
        model=model,
        data=_base_data,
        cfg=cfg,
        device=device,
        verbatim=False,
        optuna_trial=trial,
    )

    best_val = min(history["loss_val"])

    row = {
        "trial": trial.number,
        "state": "complete",
        "rmse": best_val,
        "hidden": cfg.hidden,
        "n_layers": cfg.n_layers,
        "lambda_phys": cfg.lambda_phys,
        "lambda_ic": cfg.lambda_ic,
        "lambda_bc": cfg.lambda_bc,
        "lr": cfg.lr,
        "scheduler_gamma": cfg.scheduler_gamma,
        "scheduler_step": cfg.scheduler_step,
    }

    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return best_val


def print_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Print trial progress."""
    print(
        f"  Trial {trial.number:>4} | "
        f"State: {trial.state.name:<10} | "
        f"RMSE: {f'{trial.value:.6f}' if trial.value is not None else 'pruned':>12} | "
        f"Best:  {study.best_value:.6f}"
    )


if __name__ == "__main__":
    my_study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
        sampler=optuna.samplers.TPESampler(seed=SEED),
        storage=f"sqlite:///{OUTPUT_PATH}/optuna.db",
        study_name="Fisher_KPP",
        load_if_exists=True,
    )

    my_study.optimize(
        objective,
        n_trials=N_TRIALS,
        callbacks=[print_callback],
    )

    print(f"\nBest value : {my_study.best_value:.6f}")
    print("Best params:")
    for k, v in my_study.best_params.items():
        print(f"  {k:<25} {v}")

    with open(OUTPUT_PATH / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(
            {"best_val": my_study.best_value, "best_params": my_study.best_params},
            f,
            indent=4,
        )

    df = my_study.trials_dataframe()
    df.to_csv(OUTPUT_PATH / "all_trials.csv", index=False)
    print(f"Saved to {OUTPUT_PATH}")
```

- [ ] **Step 2: Create `Fisher_KPP/scripts/optuna_tuning_plots.py`**

```python
"""
Load the best hyperparameter-tuned model, retrain it, and save plots.

Usage:
    python optuna_tuning_plots.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
from data import generate_data
from model import FCNet
from trainer import train
from utils import get_device, save_model, load_best_cfg
from plot import save_plots_from_file

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "outputs/optuna_tuning"


def main() -> None:
    """Main loop."""
    device = get_device()
    print(f"Device : {device}\n")

    best_params_path = OUTPUT_PATH / "best_params.json"
    if not best_params_path.exists():
        print(f"No best_params.json found at {best_params_path} — run optuna_tuning.py first.")
        return

    cfg = load_best_cfg(best_params_path)
    print(f"Best cfg : {cfg}")

    data = generate_data(cfg)
    model = FCNet(cfg)
    history, snapshots, best_state = train(
        model=model,
        data=data,
        cfg=cfg,
        device=device,
        label="Optuna best",
    )

    save_model(best_state, history, snapshots, cfg, OUTPUT_PATH)
    print(f"Model saved to {OUTPUT_PATH}")

    save_plots_from_file(OUTPUT_PATH)
    print(f"Plots saved to {OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify both scripts import cleanly**

Run:
```bash
cd Fisher_KPP && uv run python -c "
import matplotlib
matplotlib.use('Agg')
import sys
sys.path.insert(0, 'scripts')
import optuna_tuning, optuna_tuning_plots
print('Both optuna scripts import OK')
"
```
Expected: prints `Both optuna scripts import OK` (importing `optuna_tuning` runs its module-level `generate_data(Config())` once at import time — matches Burgers' identical pattern — so this will take a few seconds but should not error).

- [ ] **Step 4: Smoke-verify the objective function with a single trial and reduced epochs**

Run:
```bash
cd Fisher_KPP && uv run python -c "
import matplotlib
matplotlib.use('Agg')
import sys
sys.path.insert(0, 'scripts')
import optuna
from config import Config
from data import generate_data
import optuna_tuning

optuna_tuning._base_cfg = Config(n_x=100, T=1.0, n_t=20, n_epochs=30, log_every=10, patience=1000)
optuna_tuning._base_data = generate_data(optuna_tuning._base_cfg)

study = optuna.create_study(direction='minimize')
study.optimize(optuna_tuning.objective, n_trials=1)
print('OK: single trial completed, best value =', study.best_value)
"
```
Expected: prints training output for one trial, then `OK: single trial completed, best value = <float>`, no exceptions.

- [ ] **Step 5: Commit**

```bash
git add Fisher_KPP/scripts/optuna_tuning.py Fisher_KPP/scripts/optuna_tuning_plots.py
git commit -m "Add Fisher_KPP Optuna hyperparameter tuning scripts"
```

---

### Task 6: Full suite and repo-wide integration check

**Files:** none new — verification only.

- [ ] **Step 1: Run the Fisher_KPP test suite**

Run: `cd Fisher_KPP && uv run pytest tests -v`
Expected: all tests PASS, including the new `t_train`/`train_extrap`/`load_best_cfg` tests from Task 1 (no regressions in the 29 pre-existing tests).

- [ ] **Step 2: Run the repo-wide test runner**

Run: `cd /Users/imranmunir/dev/pinn-dho-burgers && uv run python run_tests.py`
Expected: `Damped_Oscillator`, `Burgers_Equation`, `Fisher_KPP` all `PASSED`; final line `ALL PROJECTS PASSED`.

- [ ] **Step 3: Verify all five new scripts import cleanly together**

Run:
```bash
cd Fisher_KPP && uv run python -c "
import matplotlib
matplotlib.use('Agg')
import sys
sys.path.insert(0, 'scripts')
import pinn_nophys, pinn_cases, pinn_blind_extrap, optuna_tuning, optuna_tuning_plots
print('All parity scripts import OK')
"
```
Expected: prints `All parity scripts import OK`.

- [ ] **Step 4: Commit (only if Steps 1-3 required fixes)**

```bash
git add -A
git commit -m "Fix issues found in Fisher_KPP parity-scripts full-suite run"
```
(Skip this step if no fixes were needed.)

---

## Self-Review Notes

- **Spec coverage:** every spec section (Optuna tuning + plots, no-physics ablation, D/r case sweep, blind extrapolation + core domain split) maps to a task above (Tasks 1-5; Task 6 is verification).
- **Placeholder scan:** no TBD/TODO; every step has complete, runnable code.
- **Type consistency:** `Config.t_train`/`train_extrap` names are used identically in `data.make_collocation` and `scripts/pinn_blind_extrap.py`. `load_best_cfg` signature (`json_path: str/Path -> Config`) matches its use in `optuna_tuning_plots.py`. All new scripts reuse the exact `save_model`/`save_plots_from_file`/`get_device` signatures already established in the core implementation — no new interfaces invented.
