"""System interface every simulated system implements (plan-reality-engine.md §3)."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

# tau_ext(t, theta, omega) -> external torque [N·m]; None means no external force.
ForceFn = Callable[[float, float, float], float]


class System:
    name: str
    state_names: list[str]
    state_units: list[str]
    param_names: list[str]
    param_units: list[str]
    nominal: dict[str, float]

    def nominal_params(self) -> np.ndarray:
        return np.array([self.nominal[k] for k in self.param_names], dtype=np.float64)

    def rhs(self, t: float, x: np.ndarray, p: np.ndarray, force_fn: ForceFn | None = None) -> np.ndarray:
        """dx/dt."""
        raise NotImplementedError

    def energy(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Total mechanical energy [J]. ``x`` may be [S] or [..., S]."""
        raise NotImplementedError

    def positions(self, x: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
        """x/y of each body [m] for playback. ``x`` may be [S] or [..., S]."""
        raise NotImplementedError

    def sample_initial_state(self, rng: np.random.Generator, regime: str) -> np.ndarray:
        raise NotImplementedError
