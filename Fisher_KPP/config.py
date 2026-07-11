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
    t_train: float = None  # end of the training window; defaults to T
    train_extrap: bool = True  # if False, collocation points are restricted to [0, t_train]

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

    def __post_init__(self):
        if self.t_train is None:
            self.t_train = self.T
