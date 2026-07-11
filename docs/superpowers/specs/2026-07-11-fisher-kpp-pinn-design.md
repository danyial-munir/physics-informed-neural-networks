# Fisher-KPP PINN — Design Spec

Date: 2026-07-11

## Goal

Add a new project `Fisher_KPP/` to this repo, solving the KPP-Fisher
reaction-diffusion equation with a Physics-Informed Neural Network, mirroring
the existing `Burgers_Equation/` and `Damped_Oscillator/` project structure
and conventions. Both forward (solve PDE) and inverse (recover D, r from
data) problems are in scope.

Reference: https://en.wikipedia.org/wiki/KPP%E2%80%93Fisher_equation

## Equation

```
u_t = D * u_xx + r * u * (1 - u)
```

Domain: x ∈ [-L, L], t ∈ [0, T].

## Initial / boundary conditions

- IC: smoothed step, `u(x,0) = 0.5 * (1 - tanh(x/eps))` — differentiable,
  avoids the non-differentiable exact step at x=0 which causes PINN
  training/gradient difficulty.
- BC: Dirichlet, `u(-L,t) = 1`, `u(L,t) = 0`.
- Analytic wave speed `c = 2*sqrt(r*D)` used as a sanity-check overlay in
  plots (KPP has no general closed-form solution outside the special
  Ablowitz-Zeppetella speed, which is out of scope for v1).

## Default parameters

- D = 1, r = 1 → wave speed c = 2.
- L, T sized so the traveling front stays inside the domain for the full
  training window (e.g. L=20, T=6) — exact values tuned during
  implementation to keep the front away from boundaries at t=T.

## Reference solution (forward validation)

Explicit finite-difference solver mirroring `Burgers_Equation/analytic.py`:
central differences for `u_xx`, explicit reaction term, CFL-checked stable
`dt` (`dt <= dx^2 / (2D)`, tightened if the reaction term requires it). Used
as ground truth for forward-PINN comparison plots and as the source for
synthetic inverse-problem observations.

## Architecture — mirrors Burgers_Equation exactly

New `Fisher_KPP/` directory, same file layout as `Burgers_Equation/`:

- `config.py` — `Config` dataclass: D, r, eps (IC smoothing width), L, T
  domain, n_x/n_t FD grid resolution, loss weights (lambda_phys, lambda_ic,
  lambda_bc), point counts (n_col, n_ic, n_bc), network shape
  (hidden, n_layers), training hyperparams (lr, lr_inverse, n_epochs,
  patience, scheduler_gamma, scheduler_step, adam betas), snapshot_epochs,
  log_every. No IC-type branching needed (KPP has one IC shape, unlike
  Burgers' multiple IC variants).
- `model.py` — `FCNet` (forward: 2 → hidden×n_layers → 1, Tanh activations,
  autograd-differentiable). `InverseFCNet`: same backbone plus two learnable
  raw parameters (`_D_raw`, `_r_raw`) exposed as positive `D_hat`, `r_hat`
  properties via softplus, softplus-inverse initialization — same pattern as
  Burgers' `nu_hat`. Plus `predict`/`predict_from_state` helpers.
- `losses.py` — `loss_data` (MSE vs noisy observations), `loss_ic`,
  `loss_bc`, `loss_physics` (residual = `u_t - D*u_xx - r*u*(1-u)`, computed
  via nested `torch.autograd.grad`), `loss_physics_inverse` (same residual
  using `model.D_hat`, `model.r_hat`).
- `analytic.py` — FD solver: `compute_stable_dt`, stepping function, top-level
  `solve_kpp_fd(cfg)` returning the (t, x, u) grid.
- `data.py` — collocation/IC/BC point sampling, mirrors
  `Burgers_Equation/data.py`.
- `trainer.py` — training loop supporting forward mode
  (physics+ic+bc losses) and inverse mode (physics_inverse+data losses),
  snapshotting best model by patience, matches Burgers' trainer structure.
- `plot.py` — solution heatmap, time-snapshot line plots vs FD reference,
  wave-speed sanity check, inverse-recovery convergence plot (D_hat, r_hat
  vs epoch).
- `utils.py` — `get_device()` etc., copied as-is from Burgers_Equation.

### Scripts (`Fisher_KPP/scripts/`) — core subset only

- `pinn_demo.py` — forward: train, plot against FD reference.
- `inversepinn_data_generation.py` — generate noisy synthetic observations
  from the FD solution.
- `inversepinn_demo.py` — inverse: train, recover D and r, plot.
- `report_plots.py` — final report figures.

Explicitly out of scope for v1 (present in Burgers_Equation but not ported):
optuna hyperparameter tuning scripts, profiling script, blind-extrapolation
script, no-physics ablation script. Can be added later following the same
pattern if needed.

### Tests (`Fisher_KPP/tests/`)

`test_data.py`, `test_smoke.py` (model forward pass, loss shapes),
`test_training.py` (short training-loop regression) — following the existing
per-project test conventions. Discovered automatically by the repo's
existing `run_tests.py` / pytest infra; no changes needed there.

## Out of scope

- Ablowitz-Zeppetella exact traveling-wave solution.
- Optuna tuning, profiling, blind-extrapolation, no-physics ablation scripts.
- Any change to `Burgers_Equation/` or `Damped_Oscillator/`.
