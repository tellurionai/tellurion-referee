"""Sanity report on a generated dataset: schema, split hygiene, labels, energy, period."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from . import integrate
from .perturbations import expected_verdict
from .systems import get_system

# name -> (shape in symbols, dtype). N trajectories, T1 = len(times), S states, P params.
# dtype None = any unicode string array.
SCHEMA = {
    "times": (("T1",), np.float64),
    "true_states": (("N", "T1", "S"), np.float32),
    "measured_states": (("N", "T1", "S"), np.float32),
    "measured_mask": (("N", "T1", "S"), np.uint8),
    "true_energy": (("N", "T1"), np.float32),
    "nominal_params": (("N", "P"), np.float32),
    "actual_params": (("N", "P"), np.float32),
    "trajectory_id": (("N",), np.int64),
    "split": (("N",), None),
    "regime": (("N",), None),
    "perturbation_type": (("N",), None),
    "perturbation_subtype": (("N",), None),
    "severity": (("N",), None),
    "onset_step": (("N",), np.int32),
    "expected_verdict": (("N",), None),
}
ENERGY_DRIFT_TOL = 1e-5   # relative; stored energy is float32
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


def schema_errors(ds: dict, meta: dict) -> list[str]:
    """Exact contract shapes and dtypes, with sizes taken from metadata and the data."""
    missing = [k for k in SCHEMA if k not in ds]
    if missing:
        return [f"missing {missing}"]
    dims = {"N": len(ds["trajectory_id"]), "T1": meta["n_steps"] + 1,
            "S": len(meta["state_names"]), "P": len(meta["param_names"])}
    errors = []
    for k, (sym, dtype) in SCHEMA.items():
        want = tuple(dims[s] for s in sym)
        if ds[k].shape != want:
            errors.append(f"{k} shape {ds[k].shape} != {want}")
        if (ds[k].dtype.kind != "U") if dtype is None else (ds[k].dtype != dtype):
            errors.append(f"{k} dtype {ds[k].dtype} != {dtype or 'str'}")
    return errors


def check_dataset(path: str | Path, include_test: bool = False) -> tuple[bool, list[str]]:
    """Sanity report. The locked test file is only read with ``include_test``."""
    from .dataset import (  # local: avoids a cycle at import time
        DEV_SPLITS,
        SPLITS,
        _paths,
        load_dataset,
    )

    meta = json.loads(_paths(path)[2].read_text())
    ds = load_dataset(path, SPLITS if include_test else DEV_SPLITS)
    system = get_system(meta["system"])
    lines, ok = [], True

    def report(name, passed, detail=""):
        nonlocal ok
        ok &= bool(passed)
        lines.append(f"[{'PASS' if passed else 'FAIL'}] {name}{': ' + detail if detail else ''}")

    errors = schema_errors(ds, meta)
    report("schema", not errors, "; ".join(errors) if errors else "all 15 arrays match contract shapes/dtypes")
    if errors:
        return ok, lines  # later checks index into these arrays

    n, t1 = ds["true_states"].shape[:2]
    lines.insert(0, f"{n} trajectories x {t1} samples, dt={meta['dt']} s, seed={meta['seed']}, "
                    f"commit={meta['git_commit'][:12]}, test {'included' if include_test else 'not loaded'}")
    report("unique trajectory_id", len(np.unique(ds["trajectory_id"])) == n)

    tv = np.isin(ds["split"], ["train", "val"])
    clean_tv = np.all((ds["regime"][tv] == "gentle") & (ds["perturbation_type"][tv] == "none"))
    report("train/val are gentle + none only", clean_tv)

    want = np.array([expected_verdict(r, p) for r, p in zip(ds["regime"], ds["perturbation_type"])])
    report("expected_verdict follows the rule", np.array_equal(want, ds["expected_verdict"]))

    none = ds["perturbation_type"] == "none"
    report("onset_step = -1 iff none", np.all((ds["onset_step"] == -1) == none))

    not_sf = ds["perturbation_type"] != "sensor_fault"
    same = np.array_equal(ds["true_states"][not_sf], ds["measured_states"][not_sf]) and np.all(ds["measured_mask"][not_sf] == 1)
    report("measured == true outside sensor_fault", same)
    nan_ok = np.array_equal(np.isnan(ds["measured_states"]), ds["measured_mask"] == 0)
    report("NaN exactly where mask = 0", nan_ok)

    # Energy drift on runs whose true physics is the ideal nominal pendulum.
    ideal = (none | ~not_sf) & (ds["nominal_params"][:, 3] == 0)
    e = ds["true_energy"][ideal].astype(np.float64)
    drift = np.abs(e - e[:, :1]).max(axis=1) / np.maximum(e[:, 0], 1e-3)
    report("energy drift (ideal runs)", drift.max() < ENERGY_DRIFT_TOL,
           f"max relative {drift.max():.2e} over {ideal.sum()} runs (tol {ENERGY_DRIFT_TOL:g})")

    g, L = system.nominal["g"], system.nominal["L"]
    expected = 2 * math.pi * math.sqrt(L / g)
    measured = small_angle_period(system)
    err = abs(measured - expected) / expected
    report("small-angle period", err < PERIOD_TOL, f"{measured:.5f} s vs 2π√(L/g) = {expected:.5f} s ({err:.2e})")

    lines.append("label counts per split (type / verdict):")
    for split in SPLITS if include_test else DEV_SPLITS:
        m = ds["split"] == split
        types = dict(Counter(ds["perturbation_type"][m].tolist()))
        verdicts = dict(Counter(ds["expected_verdict"][m].tolist()))
        lines.append(f"  {split:5s} n={m.sum():5d}  {types}  {verdicts}")
    return ok, lines
