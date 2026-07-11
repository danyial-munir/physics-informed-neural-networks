# Physics-Informed Neural Networks (PINNs)

**Author:** Danyial Munir  
**Email:** danyial.munir@studio.unibo.it  
**Last modified:** 08-July-2026

This project applies Physics-Informed Neural Networks (PINNs) to three classical problems in mathematical physics:

1. **Damped Harmonic Oscillator** — a second-order ODE
2. **Burgers' Equation** — a nonlinear PDE describing convection and diffusion
3. **Fisher-KPP Equation** — a nonlinear reaction-diffusion PDE describing traveling-front propagation

Both forward problems (predict the solution given known parameters) and inverse problems (recover unknown physical parameters from noisy observations) are implemented.

---

## Project structure

```
.
├── Damped_Oscillator/
│   ├── config.py       # Physical constants, hyperparameters, runtime settings
│   ├── data.py         # Analytic solution and data generation
│   ├── model.py        # FCNet and InverseFCNet architectures
│   ├── losses.py       # Loss functions (data, physics, initial condition)
│   ├── trainer.py      # Training loop with early stopping and snapshots
│   ├── plot.py         # All plotting functions
│   ├── utils.py        # Helper utilities (device, tensors, save/load)
│   └── scripts/        # Example scripts to train PINNs
│
├── Burgers_Equation/
│   ├── config.py       # Physical constants, hyperparameters, runtime settings
│   ├── analytic.py     # Ground truth solvers (Cole-Hopf, Euler, Lax-Wendroff)
│   ├── data.py         # Data generation (observations, collocation, validation)
│   ├── model.py        # FCNet and InverseFCNet architectures
│   ├── losses.py       # Loss functions (data, physics, IC, boundary condition)
│   ├── trainer.py      # Training loop with early stopping and snapshots
│   ├── plot.py         # All plotting functions
│   ├── utils.py        # Helper utilities (device, tensors, save/load)
│   └── scripts/        # Example scripts to train PINNs
│
└── Fisher_KPP/
    ├── config.py       # Physical constants, hyperparameters, runtime settings
    ├── analytic.py     # CFL-stable finite-difference reference solver
    ├── data.py         # Data generation (observations, collocation, validation)
    ├── model.py        # FCNet and InverseFCNet architectures
    ├── losses.py       # Loss functions (data, physics, IC, boundary condition)
    ├── trainer.py      # Training loop with early stopping and snapshots
    ├── plot.py         # All plotting functions
    ├── utils.py        # Helper utilities (device, tensors, save/load)
    └── scripts/        # Example scripts to train PINNs
```

Each module is self-contained and importable. Training experiments are written as separate scripts that import from these modules.

---

## Getting started

### Prerequisites

- Python 3.14, as defined in `.python-version` and `pyproject.toml`
- `pip` or `uv`
- A CUDA-capable GPU is supported but not required. The code automatically selects CUDA > MPS > CPU.

### Set up with uv

This is the recommended path because the repository includes `pyproject.toml` and `uv.lock`.

```bash
uv sync
```

If your environment needs an explicit writable cache location, use:

```bash
uv --cache-dir .uv-cache sync
```

Run commands through the project environment with:

```bash
uv run python main.py
uv run python Damped_Oscillator/scripts/pinn_demo.py
uv run python Burgers_Equation/scripts/pinn_demo.py
uv run python Fisher_KPP/scripts/pinn_demo.py
```

### Set up with pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run commands with:

```bash
python main.py
python Damped_Oscillator/scripts/pinn_demo.py
python Burgers_Equation/scripts/pinn_demo.py
python Fisher_KPP/scripts/pinn_demo.py
```

### Dependencies

The core dependencies are:

```text
torch
numpy
matplotlib
scikit-learn
pandas
tqdm
optuna
memory-profiler
```

### Testing

Install dev deps once:

```bash
uv sync --group dev
```

Run the complete suite — all three projects, smoke + training-loop regression, in parallel, per-test results and per-file coverage for each project, plus a pass/fail summary, all in one command:

```bash
uv run python run_tests.py
```

Run a single project's suite (with the same verbose + coverage output):

```bash
cd Damped_Oscillator && uv run python -m pytest tests -v --cov=. --cov-report=term-missing
cd Burgers_Equation && uv run python -m pytest tests -v --cov=. --cov-report=term-missing
cd Fisher_KPP && uv run python -m pytest tests -v --cov=. --cov-report=term-missing
```

Run only smoke tests (fast, no training) or only regression tests (slower, trains tiny nets for a few hundred epochs) per project:

```bash
cd Damped_Oscillator && uv run python -m pytest tests/test_smoke.py
cd Damped_Oscillator && uv run python -m pytest tests/test_training.py

cd Burgers_Equation && uv run python -m pytest tests/test_smoke.py
cd Burgers_Equation && uv run python -m pytest tests/test_training.py

cd Fisher_KPP && uv run python -m pytest tests/test_smoke.py
cd Fisher_KPP && uv run python -m pytest tests/test_training.py
```

---

## Chapter 1 — Damped Harmonic Oscillator

### Physics

The model solves the second-order linear ODE:

$$my''(t) + cy'(t) + ky(t) = 0$$

with initial conditions $y(0) = y_0$ and $y'(0) = dy_0$. The behaviour depends on the damping ratio $\zeta = c / (2\sqrt{mk})$ and natural frequency $\omega_0 = \sqrt{k/m}$:

| Regime | Condition | Solution form |
|---|---|---|
| Overdamped | $\zeta > 1$ | Sum of two real exponentials |
| Critically damped | $\zeta = 1$ | $(A + Bt)e^{-\omega_0 t}$ |
| Underdamped | $\zeta < 1$ | Exponentially decaying sinusoid |

### File descriptions

#### `config.py`
Defines the `Config` dataclass with all parameters. Physical parameters `m`, `c`, `k` are stored directly; `omega_0`, `zeta`, and `omega_d` are computed as `@property` fields. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `m`, `c`, `k` | `1.0`, `0.5`, `4.0` | Physical constants |
| `t_dom` | `6.0` | End of the training domain |
| `t_extrap` | `10.0` | End of the extrapolation domain |
| `n_obs` | `15` | Noisy observation points per epoch |
| `n_col_dom` | `500` | Collocation points in the training domain |
| `hidden` | `64` | Nodes per hidden layer |
| `n_layers` | `5` | Number of hidden layers |
| `lr` | `5e-3` | Initial learning rate |
| `lambda_phys` | `1e-2` | Weight for the physics loss term |
| `lambda_ic` | `10` | Weight for the initial condition loss term |
| `patience` | `1000` | Early stopping patience (epochs) |

#### `data.py`
Generates all data required for training:

- `analytic(t, cfg)` — closed-form solution for $y(t)$ supporting all three damping regimes.
- `make_train_observation(cfg)` — generates `n_obs` noisy observation points drawn uniformly from $(0.1,\ t_{dom})$.
- `make_validation(cfg)` — generates a dense, evenly spaced validation set on the training domain.
- `make_collocation(cfg)` — generates collocation points on the training domain and (optionally) the extrapolation domain.
- `generate_data(cfg)` — pipeline that calls all of the above and returns a single `dict`.

#### `model.py`
Two network architectures are defined:

- **`FCNet`**: maps $t \to y(t)$. Architecture: `1 → [hidden × n_layers] → 1` with `Tanh` activations and optional dropout. `Tanh` is required (not `ReLU`) because the network output is differentiated twice via `autograd`.
- **`InverseFCNet`**: same architecture as `FCNet` but adds learnable scalar parameters `zeta_hat` and `omega_0_hat` that are optimised alongside the network weights.

Helper functions:

- `predict(model, t)` — runs inference on a numpy array of times, returns a numpy array.
- `predict_from_state(state_dict, t, cfg)` — restores a saved state dict and predicts. Automatically detects whether the state dict belongs to a forward or inverse model.

#### `losses.py`
Three loss functions, each returning a scalar `torch.Tensor`:
- **`loss_data`** — MSE between predicted and observed $y$ values: $\mathcal{L}_{data} = \frac{1}{M}\sum_{j=1}^{M}(\hat{y}(t_j) - y_j)^2$
- **`loss_physics`** — mean squared ODE residual at collocation points. Both $\hat{y}'$ and $\hat{y}''$ are computed with `torch.autograd.grad` using `create_graph=True` so the second derivative can be backpropagated through: $\mathcal{L}_{physics} = \frac{1}{N}\sum_{i=1}^{N}(m\hat{y}''(t_i) + c\hat{y}'(t_i) + k\hat{y}(t_i))^2$
- **`loss_physics_inverse`** — same as above but uses `model.zeta_hat` and `model.omega_0_hat` instead of fixed `cfg.c` and `cfg.k`.
- **`loss_ic`** — squared error on both initial conditions, evaluated at $t=0$ using `autograd` for $\hat{y}'(0)$: $\mathcal{L}_{ic} = (\hat{y}(0) - y_0)^2 + (\hat{y}'(0) - dy_0)^2$

The total loss is:
$\mathcal{L}_{tot} = \mathcal{L}_{data} + \lambda_{phys}\mathcal{L}_{physics} + \lambda_{ic}\mathcal{L}_{ic}$

#### `trainer.py`
The `train()` function runs the full training loop. Key features:

- **Automatic device selection** — model and tensors are moved to the best available device.
- **Inverse mode detection** — checks for `zeta_hat` attribute; if present, uses separate learning rates for the network weights (`lr`) and the physical parameters (`lr_inverse`).
- **Randomised training data** — observation and collocation points are regenerated every epoch if `cfg.randomise_observation` / `cfg.randomise_collocation` is `True`.
- **Early stopping** — training halts when the validation loss has not improved by more than `patience_threshold` for `patience` consecutive epochs.
- **Snapshots** — the model state dict is saved at epochs listed in `cfg.snapshot_epochs`.
- **Optuna integration** — pass an `optuna.Trial` object to enable pruning of unpromising hyperparameter trials.

Returns `(history, snapshots, best_state)`:
- `history`: dict of loss curves logged every `log_every` epochs.
- `snapshots`: dict mapping epoch number to CPU state dict.
- `best_state`: state dict corresponding to the lowest validation loss seen during training.

#### `utils.py`
Shared helper functions:

- `get_device()` — returns the best available `torch.device`.
- `set_seed(seed)` — sets seeds for `random`, `numpy`, and `torch`.
- `to_tensor(arr, requires_grad, unsqueeze)` — converts a numpy array to a `(N, 1)` float32 tensor on the active device.
- `rmse(pred, true)` — computes root mean squared error.
- `save_model(best_state, history, snapshots, cfg, output_path)` — saves model weights (`.pt`), loss history (`.csv`), snapshots (`.pt`), and config (`.json`) to a folder.
- `load_model(model, folder_path)` — restores a model and all associated data from a saved folder.
- `load_best_cfg(json_path)` — loads the best Optuna trial parameters from a JSON file as a `Config` object.

---

## Chapter 2 — Burgers' Equation

### Physics

The viscid Burgers' equation is:

$u_t + u u_x = \nu u_{xx}$

where $\nu$ is the kinematic viscosity. The solution $u(x, t)$ represents a 1D velocity field. The nonlinear advection term causes wave steepening and eventual shock formation; the viscosity term prevents true discontinuities from forming.

Three initial conditions are supported: Gaussian, N-wave, and step functions. The shock formation time can be estimated analytically via the method of characteristics as $t_{shock} = -1/\min_x u_x(x, 0)$.

### File descriptions

#### `config.py`
Similar structure to the DHO config. Key additional parameters:

| Parameter | Default | Description |
|---|---|---|
| `nu` | `0.05` | Kinematic viscosity |
| `ic` | `"Gauss"` | Initial condition type |
| `L` | `15` | Spatial domain length |
| `n_x` | `1000` | Spatial grid resolution |
| `t_dom` | `5.0` | End of the training domain |
| `t_extrap` | `7.0` | End of the extrapolation domain |
| `n_col_dom` | `200` | Collocation points |
| `n_bc` | `50` | Boundary condition points per epoch |
| `lambda_phys` | `5` | Weight for physics loss |
| `lambda_ic` | `10` | Weight for IC loss |
| `lambda_bc` | `10` | Weight for boundary condition loss |

`__post_init__` automatically sets IC-specific defaults (amplitude, width, boundary values) based on the `ic` field.

Supported initial conditions:

| `ic` | Shape |
|---|---|
| `"Gauss"` | Gaussian bell curve centred at $x = L/2$ |
| `"N_wave"` | Antisymmetric linear ramp (forms a bilateral shock) |
| `"N_wave_chop"` | Half N-wave rearranged to create a single steep front |
| `"Step_up"` | Step from 0 to `height` at $x = L/2$ (rarefaction, no shock) |
| `"Step_down"` | Step from `height` to 0 at $x = L/2$ |
| `"Slope"` | Linear decrease to zero over a finite width |

#### `analytic.py`
Provides ground truth solutions via three strategies:

**Cole-Hopf transformation** (primary method, used for training data):  
The viscid Burgers' equation is transformed to the heat equation via $u = -2\nu \phi_x / \phi$. The heat equation is solved by convolving the initial condition $\phi_0$ with the Gaussian heat kernel. A padding region (800 cells for step-like ICs, 400 cells otherwise) is applied to each side before convolution to suppress boundary artefacts, then removed afterwards. The solution is computed on a full time grid and stored as a `(n_t, n_x)` array.

**Forward Euler with upwind advection** (first-order in space and time):  
Used with adaptive time-stepping via a combined CFL + von Neumann stability condition.

**Lax-Wendroff scheme** (second-order in space and time):  
Reduces numerical diffusion compared to Euler at the cost of small oscillations near sharp gradients.

Additional functions:
- `u_0(cfg)` — constructs the initial condition array; optionally applies `time_to_ic > 0` steps of Burgers evolution to produce a smoothed-out starting condition (useful for N-wave ICs with discontinuities).
- `predict_shock_time(cfg)` — estimates $t_{shock}$ from the minimum spatial gradient of $u_0$.
- `interpolate_solution(sol, t_arr, x, t, cfg)` — bilinear interpolation of the solution grid at arbitrary $(x, t)$.
- `residual(sol, t_arr, cfg)` — computes the PDE residual $|u_t + uu_x - \nu u_{xx}|$ on the solution grid using finite differences.

#### `data.py`

- `make_observation(cfg, u_grid, t_arr)` — generates `n_obs` random $(t, x)$ pairs and interpolates noisy $u$ values from the Cole-Hopf grid.
- `make_collocation(cfg)` — generates random collocation points for the training and extrapolation domains.
- `make_validation(cfg, u_grid, t_arr)` — generates a regular `(n_grid_val × n_grid_val)` meshgrid of $(t, x)$ pairs with ground truth $u$ values for validation.
- `make_bc_points(cfg)` — generates random time points along the boundary for the boundary condition loss.
- `generate_data(cfg)` — full pipeline; also stores the Cole-Hopf grid and initial condition arrays in the returned dict.

#### `model.py`

- **`FCNet`**: maps $(t, x) \to u(t, x)$. Architecture: `2 → [hidden × n_layers] → 1` with `Tanh` activations.
- **`InverseFCNet`**: same architecture but adds a learnable scalar parameter `_nu_raw`. The physical viscosity is recovered as `nu_hat = softplus(_nu_raw)`, which ensures $\hat\nu > 0$ at all times. The softplus parameterisation avoids unconstrained optimisation of a physically positive quantity. `_nu_raw` is initialised from a random softplus-inverse of a uniform draw in $[0.5, 2.0]$.

#### `losses.py`

- **`loss_data`** — MSE between predicted and observed $u$ values.
- **`loss_physics`** — mean squared PDE residual: $\mathcal{L}_{physics} = \frac{1}{N}\sum_{i=1}^N (\hat{u}_t + \hat{u}\hat{u}_x - \nu\hat{u}_{xx})^2$
All derivatives ($\hat{u}_t$, $\hat{u}_x$, $\hat{u}_{xx}$) are computed with `torch.autograd.grad`.
- **`loss_physics_inverse`** — same but uses `model.nu_hat` in place of `cfg.nu`.
- **`loss_ic`** — MSE between the predicted and true initial condition over all $x$: $\mathcal{L}_{ic} = \frac{1}{N_x}\sum_i (\hat{u}(0, x_i) - u_0(x_i))^2$
- **`loss_bc`** — MSE of predicted values at both boundaries against the Dirichlet conditions: $\mathcal{L}_{bc} = \frac{1}{N}\sum_i \left[(\hat{u}(t_i, 0) - u_{left})^2 + (\hat{u}(t_i, L) - u_{right})^2\right]$

The total loss is: $\mathcal{L}_{tot} = \mathcal{L}_{data} + \lambda_{phys}\mathcal{L}_{physics} + \lambda_{ic}\mathcal{L}_{ic} + \lambda_{bc}\mathcal{L}_{bc}$

#### `trainer.py`
Same structure as the DHO trainer with the following additions:

- **Inverse mode detection** — checks for a `nu_hat` attribute on the model.
- **Boundary condition loss** — `loss_bc` is computed every epoch (BC points are randomised if `cfg.randomise_bc_points` is `True`).
- The best model state is loaded back into the model before the function returns, so the returned model is already at its best weights.

#### `utils.py`
Identical in structure to the DHO utils. The `load_best_cfg` function reads an Optuna JSON output and populates a `Config` object, mapping the `regime` key to the `ic` field.

---

## Chapter 3 — Fisher-KPP Equation

### Physics

The [Fisher-KPP equation](https://en.wikipedia.org/wiki/KPP%E2%80%93Fisher_equation) is:

$u_t = D u_{xx} + r u (1 - u)$

where $D$ is the diffusion coefficient and $r$ is the logistic growth rate. It combines linear diffusion with logistic reaction, and is the classical model for a population (or gene, or reactant) spreading through space while growing towards a carrying capacity of $u=1$. For a step-like initial condition, the solution relaxes into a traveling wave front moving at the analytic minimum speed $c = 2\sqrt{rD}$.

### File descriptions

#### `config.py`
Same structure as the other two projects. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `D` | `1.0` | Diffusion coefficient |
| `r` | `1.0` | Logistic growth rate |
| `eps` | `0.5` | Tanh smoothing width for the initial condition |
| `L` | `20.0` | Spatial domain is $x \in [-L, L]$ |
| `T` | `6.0` | Time domain is $t \in [0, T]$ |
| `n_x` | `400` | Spatial resolution of the FD reference grid |
| `n_t` | `600` | Number of FD reference time snapshots |
| `n_col` | `2000` | PDE residual collocation points |
| `n_bc` | `100` | Boundary condition points per epoch |
| `hidden` | `64` | Nodes per hidden layer |
| `n_layers` | `6` | Number of hidden layers |
| `lambda_phys` | `1.0` | Weight for the physics loss term |
| `lambda_ic` | `10.0` | Weight for the initial condition loss term |
| `lambda_bc` | `10.0` | Weight for the boundary condition loss term |
| `patience` | `4000` | Early stopping patience (epochs) |
| `t_train` | `T` | End of the collocation-sampling window; defaults to `T` (full domain) unless overridden |
| `train_extrap` | `True` | If `False`, PDE residual collocation points are restricted to `[0, t_train]` — used by the blind-extrapolation experiment |

The initial condition is a smoothed step, $u(x, 0) = 0.5(1 - \tanh(x/\varepsilon))$, avoiding the non-differentiable sharp step that would otherwise destabilise the physics-loss gradient at $x=0$. Boundary conditions are Dirichlet: $u(-L, t) = 1$, $u(L, t) = 0$.

#### `analytic.py`
Provides the ground truth reference used for both forward-PINN validation and inverse-problem synthetic observations:

- `u_0(cfg)` — the smoothed step initial condition.
- `wave_speed(cfg)` — the analytic KPP traveling-wave speed $c = 2\sqrt{rD}$, used only as a plot sanity-check overlay (Fisher-KPP has no general closed-form solution outside this special-speed case).
- `compute_stable_dt(cfg)` — CFL-stable explicit time step, $dt \le dx^2 / (2D)$, per the von Neumann diffusion stability limit. The bounded reaction term does not tighten this limit for the parameter ranges used here.
- `fd_step(u, dt, cfg)` — one explicit-Euler finite-difference step: central differences for $u_{xx}$, explicit reaction term $ru(1-u)$, Dirichlet boundaries held fixed.
- `solve_kpp_fd(cfg)` — marches `fd_step` from $u_0$ to $t=T$, returning a `(n_snapshots, n_x)` solution grid and its time array.
- `interpolate_solution(u_grid, t_arr, x, t, cfg)` — bilinear interpolation of the FD grid at arbitrary $(x, t)$.

#### `data.py`

- `make_observation(cfg, u_grid, t_arr)` — generates `n_obs` random $(t, x)$ pairs and interpolates noisy $u$ values from the FD grid.
- `make_collocation(cfg)` — generates random PDE residual collocation points over the full domain, or restricted to `[0, cfg.t_train]` when `cfg.train_extrap` is `False`.
- `make_bc_points(cfg)` — generates random time points for the boundary condition loss.
- `make_validation(cfg, u_grid, t_arr)` — generates a regular `(n_grid_val × n_grid_val)` meshgrid with ground truth $u$ values for validation.
- `generate_data(cfg)` — full pipeline; also stores the FD grid and initial condition arrays in the returned dict.

#### `model.py`

- **`FCNet`**: maps $(t, x) \to u(t, x)$. Architecture: `2 → [hidden × n_layers] → 1` with `Tanh` activations.
- **`InverseFCNet`**: same architecture but adds two learnable scalar parameters, `_D_raw` and `_r_raw`. The physical parameters are recovered as `D_hat = softplus(_D_raw)` and `r_hat = softplus(_r_raw)`, ensuring both stay positive throughout optimisation.

#### `losses.py`

- **`loss_data`** — MSE between predicted and observed $u$ values.
- **`loss_physics`** — mean squared PDE residual: $\mathcal{L}_{physics} = \frac{1}{N}\sum_{i=1}^N (\hat{u}_t - D\hat{u}_{xx} - r\hat{u}(1-\hat{u}))^2$, with all derivatives computed via `torch.autograd.grad`.
- **`loss_physics_inverse`** — same but uses `model.D_hat` and `model.r_hat` in place of `cfg.D` and `cfg.r`.
- **`loss_ic`** — MSE between the predicted and true initial condition over all $x$.
- **`loss_bc`** — MSE of predicted values at both boundaries against the Dirichlet conditions.

The total loss is: $\mathcal{L}_{tot} = \mathcal{L}_{data} + \lambda_{phys}\mathcal{L}_{physics} + \lambda_{ic}\mathcal{L}_{ic} + \lambda_{bc}\mathcal{L}_{bc}$

#### `trainer.py`
Same structure as the other two trainers — inverse mode is detected via a `D_hat` attribute on the model, separate learning rates (`lr` for the network, `lr_inverse` for `_D_raw`/`_r_raw`) are used in inverse mode, and an optional `optuna_trial` argument enables mid-training pruning (see [Hyperparameter tuning with Optuna](#hyperparameter-tuning-with-optuna)).

#### `plot.py`
- `plot_solution_heatmap` — side-by-side heatmap of the FD reference and the PINN prediction over the full $(x, t)$ domain. Correctly reconstructs either `FCNet` or `InverseFCNet` from a saved state dict.
- `plot_snapshots` — FD reference vs. PINN prediction at fixed times, with the analytic wave-speed front position marked.
- `plot_losses` — training loss curves on a log scale.
- `plot_predicted_parameter_convergence` — $\hat{D}$, $\hat{r}$ vs. epoch against the ground-truth values (inverse mode).
- `save_plots_from_file` — loads a saved run and regenerates all plots.

#### `utils.py`
Identical in structure to the other two projects, including `load_best_cfg(json_path)` for restoring an Optuna best-trial config.

## Running the demo scripts

All three projects ship ready-to-run demo scripts under their respective `scripts/` folders. Use `uv run` when you set up the project with uv, or activate `.venv` and use `python` when you set it up with pip.

### Command format

```bash
uv run python path/to/script.py [options]
python path/to/script.py [options]
```

The examples below use `uv run`. To use pip instead, remove the `uv run` prefix after activating your virtual environment.

---

### Damped Harmonic Oscillator

#### Forward problem

```bash
uv run python Damped_Oscillator/scripts/pinn_demo.py
uv run python Damped_Oscillator/scripts/pinn_demo.py --underdamped
uv run python Damped_Oscillator/scripts/pinn_demo.py --zeta 0.5 --omega_0 3.0
uv run python Damped_Oscillator/scripts/pinn_demo.py --overdamped --output outputs/my_run
```

#### Inverse problem

```bash
uv run python Damped_Oscillator/scripts/inversepinn_demo.py
uv run python Damped_Oscillator/scripts/inversepinn_demo.py --underdamped
uv run python Damped_Oscillator/scripts/inversepinn_demo.py --zeta 0.1 --omega_0 2.0
uv run python Damped_Oscillator/scripts/inversepinn_demo.py --overdamped --output outputs/my_run
```

#### Shared CLI arguments — Damped Harmonic Oscillator

| Argument | Default | Description |
|---|---|---|
| `--zeta` | — | Fix damping ratio ζ; skips randomisation |
| `--omega_0` | — | Fix natural frequency ω₀; skips randomisation |
| `--underdamped` | — | Use underdamped regime (ζ < 1). Forward: fixed at 0.125; inverse: randomised in [0.02, 0.2] |
| `--critically-damped` | — | Use critically damped case (ζ = 1) |
| `--overdamped` | — | Use overdamped regime (ζ > 1). Forward: fixed at 1.5; inverse: randomised in [1.5, 2.0] |
| `--show` | `False` | Display plots interactively after training |
| `--output` | `outputs/<script>_<regime>` | Override the output directory |

If no damping flag is given, the forward script defaults to ζ = 0.125 and the inverse script randomises in the underdamped regime.

---

### Burgers' Equation

#### Forward problem

```bash
uv run python Burgers_Equation/scripts/pinn_demo.py
uv run python Burgers_Equation/scripts/pinn_demo.py --ic N_wave --nu 0.05
uv run python Burgers_Equation/scripts/pinn_demo.py --ic Step_up --output outputs/my_run
```

#### Inverse problem

```bash
uv run python Burgers_Equation/scripts/inversepinn_demo.py
uv run python Burgers_Equation/scripts/inversepinn_demo.py --nu 0.3 --ic Gauss
uv run python Burgers_Equation/scripts/inversepinn_demo.py --nu-regime low --output outputs/inverse_low
```

#### Shared CLI arguments — Burgers' Equation

| Argument | Default | Description |
|---|---|---|
| `--ic` | `Gauss` | Initial condition type: `Gauss`, `N_wave`, or `Step_up` |
| `--nu` | — | Fix viscosity to a specific value; skips randomisation |
| `--nu-regime` | `high` | Viscosity regime to sample from when `--nu` is not set: `low` (0.005–0.015) or `high` (0.1–0.5). Inverse scripts only. |
| `--show` | `False` | Display plots interactively after training |
| `--output` | Forward: `outputs/pinn_<ic>`; inverse: `outputs/inversepinn_<ic>_<nu-regime>` or `outputs/inversepinn_<ic>_nu_<value>` when `--nu` is fixed | Override the output directory |

---

### Fisher-KPP Equation

#### Forward problem

```bash
uv run python Fisher_KPP/scripts/pinn_demo.py
uv run python Fisher_KPP/scripts/pinn_demo.py --D 0.5 --r 2.0
uv run python Fisher_KPP/scripts/pinn_demo.py --output outputs/my_run
```

#### Inverse problem

```bash
uv run python Fisher_KPP/scripts/inversepinn_demo.py
uv run python Fisher_KPP/scripts/inversepinn_demo.py --D 0.5 --r 2.0
uv run python Fisher_KPP/scripts/inversepinn_demo.py --output outputs/my_run
```

#### Shared CLI arguments — Fisher-KPP Equation

| Argument | Default | Description |
|---|---|---|
| `--D` | `1.0` | Diffusion coefficient |
| `--r` | `1.0` | Logistic growth rate |
| `--show` | `False` | Display plots interactively after training |
| `--output` | `outputs/pinn_D<D>_r<r>` (forward) or `outputs/inversepinn_D<D>_r<r>` (inverse) | Override the output directory |

Continuous true/predicted `(D, r)` pairs for the inverse problem can be generated with:

```bash
uv run python Fisher_KPP/scripts/inversepinn_data_generation.py
```

This runs until `Ctrl+C`, randomising $D, r$ each iteration and appending results to `outputs/parameter_estimation/D_r_pairs.csv`.

#### Additional experiment scripts

| Script | Purpose | Output |
|---|---|---|
| `pinn_nophys.py` | No-physics ablation — trains a plain data-fitting network (`use_physics=False, use_ic=False, use_bc=False, use_data=True`) as a baseline to compare against the physics-informed forward run | `outputs/D1_r1_ML/` |
| `pinn_cases.py` | Sweeps `D ∈ {0.5, 1.0, 2.0}` × `r ∈ {0.5, 1.0, 2.0}`, training a forward PINN for each combination | `outputs/D{D}_r{r}/` per case |
| `pinn_blind_extrap.py` | Trains with collocation points restricted to `t_train=3.0` (half of `T=6.0`, via `train_extrap=False`), then plots the prediction across the full domain to reveal how well the traveling front generalises past its training window | `outputs/D1_r1_extrapblind/` |
| `optuna_tuning.py` | Optuna TPE hyperparameter search over `hidden`, `n_layers`, `lambda_phys`, `lambda_ic`, `lambda_bc`, `lr`, `scheduler_gamma`, `scheduler_step` (fixed `D=1.0, r=1.0` — Fisher-KPP has no IC-regime axis to loop over, unlike Burgers') | `outputs/optuna_tuning/` |
| `optuna_tuning_plots.py` | Loads `best_params.json` from the Optuna search, retrains with those hyperparameters, saves plots | `outputs/optuna_tuning/` |

```bash
uv run python Fisher_KPP/scripts/pinn_nophys.py
uv run python Fisher_KPP/scripts/pinn_cases.py
uv run python Fisher_KPP/scripts/pinn_blind_extrap.py
uv run python Fisher_KPP/scripts/optuna_tuning.py
uv run python Fisher_KPP/scripts/optuna_tuning_plots.py
```

---

### Output directory layout

```
outputs/pinn_underdamped/
├── best_model.pt
├── history.csv
├── snapshots.pt
└── config.json
```

Plots are written alongside these files.
---

## Training a model

For more customisation, it is also possible to create your own scripts using the provided code instead of using the demo files. A minimal training script follows this pattern:

```python
from config import Config
from data import generate_data
from model import FCNet
from trainer import train
from utils import get_device, save_model

device = get_device()
cfg    = Config(nu=0.05, ic="Gauss", n_epochs=20000)
data   = generate_data(cfg)
model  = FCNet(cfg)

history, snapshots, best_state = train(model, data, cfg, device)

save_model(best_state, history, snapshots, cfg, output_path="outputs/my_run")
```

For the inverse problem, replace `FCNet` with `InverseFCNet` and set `cfg.use_data = True`. The trainer detects the model type automatically and uses a separate learning rate (`cfg.lr_inverse`) for the physical parameter.

---

## Saving and loading

`save_model` writes four files to `output_path/`:

| File | Contents |
|---|---|
| `best_model.pt` | State dict with the lowest validation loss |
| `history.csv` | Loss curves, one row per `log_every` epochs |
| `snapshots.pt` | State dicts at each `snapshot_epochs` checkpoint |
| `config.json` | Full `Config` as JSON for reproducibility |

`load_model(model, folder_path)` restores all four. `save_plots_from_file(folder_path)` regenerates ground truth data and saves all plots without requiring a re-run of training.

---

## Hyperparameter tuning with Optuna

All three projects support Optuna-based hyperparameter search. Pass an `optuna.Trial` to `train()` to enable pruning. A minimal study looks like:

```python
import optuna
from config import Config
from data import generate_data
from model import FCNet
from trainer import train
from utils import get_device

def objective(trial):
    cfg = Config(
        hidden        = trial.suggest_int("hidden", 16, 128, step=16),
        n_layers      = trial.suggest_int("n_layers", 2, 6),
        lr            = trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        lambda_phys   = trial.suggest_float("lambda_phys", 1e-3, 10, log=True),
    )
    data   = generate_data(cfg)
    model  = FCNet(cfg)
    device = get_device()
    history, _, _ = train(model, data, cfg, device, verbatim=False, optuna_trial=trial)
    return min(history["loss_val"])

study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100)
```

---
