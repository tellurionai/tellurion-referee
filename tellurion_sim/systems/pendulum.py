"""Single pendulum with optional viscous damping and an external torque.

    dtheta/dt = omega
    domega/dt = -(g/L) sin(theta) - (b/(m L^2)) omega + tau_ext(t, theta, omega)/(m L^2)
    E = 0.5 m L^2 omega^2 + m g L (1 - cos theta)

theta = 0 hangs straight down, positive counter-clockwise. Pivot at (0, 0);
bob at x = L sin(theta), y = -L cos(theta).
"""
from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from .base import ForceFn, System

REGIMES = ("gentle", "large", "rotating")


class Pendulum(System):
    name = "pendulum"
    state_names: ClassVar[list[str]] = ["theta", "omega"]
    state_units: ClassVar[list[str]] = ["rad", "rad/s"]
    param_names: ClassVar[list[str]] = ["g", "L", "m", "b"]
    param_units: ClassVar[list[str]] = ["m/s^2", "m", "kg", "N·m·s/rad"]
    nominal: ClassVar[dict[str, float]] = {"g": 9.81, "L": 1.0, "m": 1.0, "b": 0.0}

    def dataset_errors(self, dataset: dict) -> list[str]:
        """Pendulum parameter bounds and the independent, healthy omega sensor."""
        errors = []
        positive = [self.param_names.index(name) for name in ("g", "L", "m")]
        damping = self.param_names.index("b")
        for key in ("nominal_params", "actual_params"):
            if not np.all(dataset[key][:, positive] > 0):
                errors.append(f"{key}: g, L and m must be positive")
            if not np.all(dataset[key][:, damping] >= 0):
                errors.append(f"{key}: damping must be nonnegative")
        omega = self.state_names.index("omega")
        if not np.array_equal(dataset["measured_states"][..., omega], dataset["true_states"][..., omega]):
            errors.append("measured omega must always equal true omega")
        if not np.all(dataset["measured_mask"][..., omega] == 1):
            errors.append("measured omega must always be valid")
        return errors

    def rhs(self, t: float, x: np.ndarray, p: np.ndarray, force_fn: ForceFn | None = None) -> np.ndarray:
        g, L, m, b = p
        theta, omega = x
        inertia = m * L * L
        domega = -(g / L) * math.sin(theta) - (b / inertia) * omega
        if force_fn is not None:
            domega += force_fn(t, theta, omega) / inertia
        return np.array([omega, domega])

    def energy(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        g, L, m, _ = p
        theta, omega = x[..., 0], x[..., 1]
        return 0.5 * m * L * L * omega**2 + m * g * L * (1.0 - np.cos(theta))

    def positions(self, x: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
        L = p[1]
        theta = x[..., 0]
        return {"bob_x": L * np.sin(theta), "bob_y": -L * np.cos(theta)}

    def sample_initial_state(self, rng: np.random.Generator, regime: str) -> np.ndarray:
        """Initial (theta0 [rad], omega0 [rad/s]) for a regime, under nominal params.

        gentle:   |theta0| <= 0.6, |omega0| <= 0.8
        large:    |theta0| in [1.5, 2.8], |omega0| <= 0.5 (stays below the top: E < 2 m g L)
        rotating: E / (2 m g L) in [1.1, 1.6], i.e. enough energy to go over the top
        """
        g, L, m = self.nominal["g"], self.nominal["L"], self.nominal["m"]
        if regime == "gentle":
            return np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.8, 0.8)])
        if regime == "large":
            theta0 = rng.choice([-1.0, 1.0]) * rng.uniform(1.5, 2.8)
            return np.array([theta0, rng.uniform(-0.5, 0.5)])
        if regime == "rotating":
            theta0 = rng.uniform(-math.pi, math.pi)
            e_total = rng.uniform(1.1, 1.6) * 2.0 * m * g * L
            kinetic = e_total - m * g * L * (1.0 - math.cos(theta0))
            omega0 = rng.choice([-1.0, 1.0]) * math.sqrt(2.0 * kinetic / (m * L * L))
            return np.array([theta0, omega0])
        raise ValueError(f"unknown regime {regime!r}; known: {REGIMES}")
