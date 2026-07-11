# Fisher_KPP Parity Scripts — Design Spec

Date: 2026-07-11

## Goal

Bring `Fisher_KPP/` closer to `Burgers_Equation/`'s script coverage by adding
four analysis/demo scripts: Optuna hyperparameter tuning, a no-physics
ablation, a (D, r) case sweep, and a blind-extrapolation experiment. This is
an enhancement pass on top of the already-complete core implementation
(config, model, losses, data, trainer, plot, core demo scripts — all on
branch `feat/fisher-kpp-with-pinn`).

## Scope

In scope:
1. `scripts/optuna_tuning.py` + `scripts/optuna_tuning_plots.py`
2. `scripts/pinn_nophys.py`
3. `scripts/pinn_cases.py`
4. `scripts/pinn_blind_extrap.py`, plus the core train/extrapolation domain
   split it depends on.

Out of scope: `pinn_profiling.py`, `pinn_hyperparameters.py`,
`pinn_data_included.py` (deferred — lower priority, not requested).

## 1. Optuna tuning

`scripts/optuna_tuning.py`: TPE search, mirrors
`Burgers_Equation/scripts/optuna_tuning.py` structurally.

- Search space: `hidden` (16–128, step 16), `n_layers` (2–10),
  `lambda_phys` (1e-1–1e2, log), `lambda_ic` (1e1–1e3, log), `lambda_bc`
  (1e1–1e3, log), `lr` (1e-4–1e-2, log), `scheduler_gamma` (0.3–0.9),
  `scheduler_step` (1000–5000, step 1000).
- Fixed `D=1.0, r=1.0` (Fisher_KPP has no IC-regime axis analogous to
  Burgers' Gauss/N_wave/Step, so this is a single study, not a loop over
  regimes).
- `objective(trial)`: builds `Config`, trains an `FCNet` with
  `use_data=True`, returns `min(history["loss_val"])`.
- `optuna.create_study(direction="minimize", pruner=MedianPruner(n_warmup_steps=20), sampler=TPESampler(seed=SEED), storage="sqlite:///.../optuna.db", study_name="Fisher_KPP", load_if_exists=True)`.
- Writes `outputs/optuna_tuning/optuna_results.csv` (per-trial row),
  `best_params.json`, `all_trials.csv`.

`scripts/optuna_tuning_plots.py`: loads `best_params.json` via
`utils.load_best_cfg`, retrains with those params, saves plots via
`save_plots_from_file`. No regime loop (single `best_params.json`, not
per-IC like Burgers).

## 2. No-physics ablation

`scripts/pinn_nophys.py`: `Config(D=1.0, r=1.0, use_data=True, use_physics=False, use_ic=False, use_bc=False)`,
trains a plain data-fitting `FCNet` (no PDE knowledge at all), saves to
`outputs/D1_r1_ML/`. Purpose: a direct before/after comparison against the
physics-informed forward run — same plotting pipeline
(`save_plots_from_file`), so the two runs' `snapshots.png` are visually
comparable.

## 3. D/r case sweep

`scripts/pinn_cases.py`: loops over `D_values = [0.5, 1.0, 2.0]`,
`r_values = [0.5, 1.0, 2.0]`, trains a forward `FCNet` for each combination,
saves each to `outputs/D{D}_r{r}/`. No shock-time logic needed (that's a
Burgers-specific concept); Fisher_KPP's domain is already sized for the
default wave speed, so cases just vary `D`/`r` directly.

## 4. Blind extrapolation

### Core change: train/extrapolation domain split

Fisher_KPP currently samples collocation points across the entire
`[0, T]`. To test whether the PINN generalizes the traveling front beyond
its training window, add a domain split mirroring Burgers' `t_dom`/`t_extrap`:

- `Config`: add `t_train: float = None` (defaults to `T` in
  `__post_init__` if unset, so existing forward/inverse behavior is
  unchanged) and `train_extrap: bool = True`.
- `data.make_collocation(cfg)`: when `train_extrap` is `True` (default),
  behavior is unchanged — collocation points span `[0, T]`. When `False`,
  collocation points are restricted to `[0, t_train]`.
- IC, BC, and validation point generation are unaffected by this split —
  they still span the full `[0, T]`, so `loss_val` and the saved plots show
  the model's error growing past `t_train` when `train_extrap=False`.
- `trainer.train`: no change needed — it already calls
  `make_collocation(cfg)` and respects whatever range it returns.

### `scripts/pinn_blind_extrap.py`

`Config(D=1.0, r=1.0, t_train=T/2, train_extrap=False)`, trains, saves to
`outputs/D1_r1_extrapblind/`, plots prediction vs FD reference across the
*full* `[0, T]` domain via the existing `save_plots_from_file` — the
front-position error past `t_train` is the interesting result, visible in
`snapshots.png`.

## Testing

- `Config` test: `t_train` defaults to `T` when unset; explicit `t_train`
  value is preserved.
- `make_collocation` test: with `train_extrap=False` and a given `t_train`,
  all returned `t_col` values are `<= t_train`; with `train_extrap=True`
  (existing default), unchanged behavior (values may exceed `t_train`).
- No new tests for the scripts themselves (matches existing convention —
  Burgers' equivalent scripts have no dedicated tests either); each script
  is smoke-verified by running a short/reduced-epoch invocation once during
  implementation.

## Out of scope

- `pinn_profiling.py`, `pinn_hyperparameters.py`, `pinn_data_included.py`.
- Any change to `Damped_Oscillator/` or `Burgers_Equation/`.
- Any change to the already-implemented core Fisher_KPP modules beyond the
  `t_train`/`train_extrap` addition described above.
