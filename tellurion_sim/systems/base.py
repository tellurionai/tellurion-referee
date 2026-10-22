"""System interface every simulated system implements (plan-reality-engine.md §3)."""
from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import numpy as np

# tau_ext(t, theta, omega) -> external torque [N·m]; None means no external force.
ForceFn = Callable[[float, float, float], float]


class System:
    name: ClassVar[str]
    state_names: ClassVar[list[str]]
    state_units: ClassVar[list[str]]
    param_names: ClassVar[list[str]]
    param_units: ClassVar[list[str]]
    nominal: ClassVar[dict[str, float]]

    def nominal_params(self) -> np.ndarray:
        return np.array([self.nominal[k] for k in self.param_names], dtype=np.float64)

    def dataset_errors(self, dataset: dict) -> list[str]:
        """System-specific invariants, called after the shared schema checks pass.

        Override for physical parameter bounds and sensor-channel assumptions.
        The base contract makes no assumptions about state or parameter order.
        """
        return []

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
