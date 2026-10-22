"""Perturbations ("knobs") and labels (plan-reality-engine.md §4).

- param_shift  -> model_error:      reality's params differ from nominal after onset
- sensor_fault -> instrument_fault: only the measured theta changes; truth untouched
- unseen_force -> discovery:        an extra torque tau_ext not in the nominal equations
- none         -> trust (model_error outside the gentle training regime)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .systems.base import ForceFn

PERTURBATION_TYPES = ("none", "param_shift", "sensor_fault", "unseen_force")
SUBTYPES = {
    "none": ("none",),
    "param_shift": ("length", "gravity", "damping"),
    "sensor_fault": ("bias", "drift", "noise", "scale", "stuck", "dropout"),
    "unseen_force": ("constant_torque", "magnet", "periodic_drive", "impulse"),
}
SEVERITIES = ("low", "med", "high")
VERDICTS = ("trust", "model_error", "instrument_fault", "discovery")
TRAINING_REGIME = "gentle"


def expected_verdict(regime: str, perturbation_type: str) -> str:
    """The one place the label rule lives (§4.4)."""
    if perturbation_type == "none":
        return "trust" if regime == TRAINING_REGIME else "model_error"
    return {"param_shift": "model_error", "sensor_fault": "instrument_fault",
            "unseen_force": "discovery"}[perturbation_type]


@dataclass(frozen=True)
class Perturbation:
    type: str = "none"
    subtype: str = "none"
    severity: str = "none"
    onset_step: int = -1
    strength: float = 0.0  # numeric value; units depend on subtype (see configs/pendulum.yaml)

    @classmethod
    def from_config(cls, cfg: dict, type: str, subtype: str, severity: str, onset_step: int) -> Perturbation:
        if type == "none":
            return cls()
        if subtype not in SUBTYPES[type]:
            raise ValueError(f"unknown {type} subtype {subtype!r}; known: {SUBTYPES[type]}")
        strength = 0.0 if severity == "none" else float(cfg[type][subtype][severity])
        return cls(type, subtype, severity, onset_step, strength)


# --- param_shift -------------------------------------------------------------

def shifted_params(nominal: np.ndarray, pert: Perturbation, param_names: list[str]) -> np.ndarray:
    """Params reality uses after onset. Zero strength returns nominal unchanged."""
    p = nominal.copy()
    if pert.type != "param_shift" or pert.strength == 0.0:
        return p
    i = {"length": "L", "gravity": "g", "damping": "b"}[pert.subtype]
    j = param_names.index(i)
    p[j] = pert.strength if pert.subtype == "damping" else p[j] * pert.strength
    return p


# --- unseen_force ------------------------------------------------------------

def force_fn(pert: Perturbation, cfg: dict, t_onset: float) -> ForceFn | None:
    """tau_ext(t, theta, omega) [N·m] active from onset, or None."""
    if pert.type != "unseen_force" or pert.subtype == "impulse":
        return None
    s = pert.strength
    if pert.subtype == "constant_torque":
        return lambda t, th, om: s
    if pert.subtype == "magnet":
        phi_m = float(cfg["magnet_phi_m"])
        return lambda t, th, om: s * math.sin(phi_m - th)
    if pert.subtype == "periodic_drive":
        w = float(cfg["periodic_drive_w"])
        return lambda t, th, om: s * math.cos(w * (t - t_onset))
    raise ValueError(pert.subtype)


def impulse_fn(pert: Perturbation):
    """on_onset(x, p) applying an angular impulse J [N·m·s]: omega += J / (m L^2)."""
    if pert.type != "unseen_force" or pert.subtype != "impulse":
        return None

    def kick(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        _, L, m, _ = p
        return np.array([x[0], x[1] + pert.strength / (m * L * L)])
    return kick


# --- sensor_fault ------------------------------------------------------------

def apply_sensor_fault(true_states: np.ndarray, pert: Perturbation, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(measured [T+1, S], mask [T+1, S] uint8). Never modifies ``true_states``.

    Faults corrupt theta (channel 0) only, from onset on.
    """
    measured = true_states.copy()
    mask = np.ones(true_states.shape, dtype=np.uint8)
    if pert.type != "sensor_fault":
        return measured, mask

    k0, s = pert.onset_step, pert.strength
    theta = measured[k0:, 0]  # view
    n = len(theta)
    if pert.subtype == "bias":
        theta += s
    elif pert.subtype == "drift":
        theta += s * np.arange(n) / max(n - 1, 1)
    elif pert.subtype == "noise":
        theta += rng.normal(0.0, 1.0, n) * s
    elif pert.subtype == "scale":
        theta *= 1.0 + s
    elif pert.subtype == "stuck":
        if s != 0.0:
            theta[:] = theta[0]
    elif pert.subtype == "dropout":
        dropped = rng.random(n) < s
        theta[dropped] = np.nan
        mask[k0:, 0][dropped] = 0
    else:
        raise ValueError(pert.subtype)
    return measured, mask
