"""Sanity report on a generated dataset: schema, split hygiene, labels, energy, period."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np

from . import integrate
from .contract import SCHEMA, schema_errors
from .dataset import (
    DEV_SPLITS,
    SPLITS,
    _archive_errors,
    _cross_archive_errors,
    _load_metadata,
    _read_archives,
)
from .systems import get_system

__all__ = ["SCHEMA", "check_dataset", "schema_errors", "small_angle_period"]

ENERGY_DRIFT_TOL = 1e-5  # relative; stored energy is float32
PERIOD_TOL = 1e-3


def small_angle_period(system, dt: float = 0.01, n_steps: int = 1000, theta0: float = 0.05) -> float:
    """Measured period [s] of a theta0 = 0.05 rad swing, from upward zero crossings."""
    p = system.nominal_params()
    times = integrate.time_grid(dt, n_steps)
    x, _ = integrate.simulate(system, np.array([theta0, 0.0]), times, p)
    th = x[:, 0]
    k = np.where((th[:-1] < 0) & (th[1:] >= 0))[0]
    crossings = times[k] - th[k] * dt / (th[k + 1] - th[k])
    return float(np.mean(np.diff(crossings)))


def check_dataset(path: str | Path, include_test: bool = False) -> tuple[bool, list[str]]:
    """Report shared contract diagnostics and physics checks, retaining provenance.

    Raw archives are used here so invalid data remains available for diagnostics.
    The public loader still rejects every contract violation before returning data.
    """
    try:
        meta = _load_metadata(path)
    except (OSError, ValueError) as error:
        return False, [f"dataset={path}; provenance unavailable", f"[FAIL] metadata: {error}"]

    archives, failures = [], []
    try:
        for file, allowed, part in _read_archives(path, SPLITS if include_test else DEV_SPLITS):
            archives.append(part)
            failures.extend(f"{file.name}: {error}" for error in _archive_errors(part, meta, allowed))
    except (OSError, ValueError) as error:
        failures.append(f"archive could not be read: {error}")
    if not failures:
        failures.extend(_cross_archive_errors(archives))

    def vector(part, key):
        value = part.get(key)
        return value if isinstance(value, np.ndarray) and value.ndim == 1 else None

    counts = [vector(part, "trajectory_id") for part in archives]
    n = sum(len(ids) for ids in counts if ids is not None)
    lines = [
        (
            f"{n} readable trajectories x {meta['n_steps'] + 1} samples, dt={meta['dt']} s, "
            f"seed={meta.get('seed', 'unknown')}, commit={str(meta.get('git_commit', 'unknown'))[:12]}, "
            f"test {'requested' if include_test else 'not loaded'}"
        )
    ]
    lines.extend(f"[FAIL] contract: {error}" for error in failures)
    if not failures:
        lines.append("[PASS] contract: required arrays and shared/system invariants")

    # Count whatever label vectors remain readable, even if other fields are invalid.
    lines.append("label counts per split (type / verdict):")
    for split in SPLITS if include_test else DEV_SPLITS:
        total, types, verdicts = 0, Counter(), Counter()
        for part in archives:
            labels = vector(part, "split")
            if labels is None or labels.dtype.kind != "U":
                continue
            keep = labels == split
            total += int(keep.sum())
            for key, counter in (("perturbation_type", types), ("expected_verdict", verdicts)):
                values = vector(part, key)
                if values is not None and values.dtype.kind == "U" and values.shape == labels.shape:
                    counter.update(values[keep].tolist())
        lines.append(f"  {split:5s} n={total:5d}  {dict(types)}  {dict(verdicts)}")
    if failures:
        return False, lines

    ds = {
        key: archives[0][key] if key == "times" else np.concatenate([part[key] for part in archives])
        for key in SCHEMA
    }
    system = get_system(meta["system"])
    if system.name != "pendulum":
        lines.append(f"[SKIP] physics checks: no checks defined for {system.name}")
        return True, lines
    ok = True

    def report(name, passed, detail=""):
        nonlocal ok
        ok &= bool(passed)
        lines.append(f"[{'PASS' if passed else 'FAIL'}] {name}{': ' + detail if detail else ''}")

    none = ds["perturbation_type"] == "none"
    # Energy drift on runs whose true physics is the ideal nominal pendulum.
    damping = system.param_names.index("b")
    ideal = (none | (ds["perturbation_type"] == "sensor_fault")) & (ds["nominal_params"][:, damping] == 0)
    e = ds["true_energy"][ideal].astype(np.float64)
    drift = np.abs(e - e[:, :1]).max(axis=1) / np.maximum(e[:, 0], 1e-3)
    if drift.size:
        report(
            "energy drift (ideal runs)",
            drift.max() < ENERGY_DRIFT_TOL,
            f"max relative {drift.max():.2e} over {ideal.sum()} runs (tol {ENERGY_DRIFT_TOL:g})",
        )
    else:
        lines.append("[SKIP] energy drift (ideal runs): no ideal trajectories")

    g, L = system.nominal["g"], system.nominal["L"]
    expected = 2 * math.pi * math.sqrt(L / g)
    measured = small_angle_period(system)
    err = abs(measured - expected) / expected
    report(
        "small-angle period", err < PERIOD_TOL, f"{measured:.5f} s vs 2π√(L/g) = {expected:.5f} s ({err:.2e})"
    )

    return ok, lines
