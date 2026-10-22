"""Executable dataset contract; see docs/DATA_CONTRACT.md.

Validation uses stored observations and labels, without rerunning the physics.
It never opens files or decides whether a locked evaluation is authorized.
"""

from __future__ import annotations

import numpy as np

from .perturbations import (
    PERTURBATION_TYPES,
    SEVERITIES,
    SUBTYPES,
    VERDICTS,
    expected_verdict,
)
from .systems import get_system
from .systems.pendulum import REGIMES

SPLITS = ("train", "val", "cal", "test")
DEV_SPLITS = ("train", "val", "cal")
LOCKED_SPLIT = "test"

# N trajectories, T1 samples, S state channels, P parameters; None means Unicode.
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


def metadata_errors(meta: dict) -> list[str]:
    """Validate the metadata needed to interpret array shapes and timing."""
    if not isinstance(meta, dict):
        return ["metadata must be a JSON object"]
    required = {"system", "n_steps", "dt", "state_names", "param_names"}
    missing = required - meta.keys()
    if missing:
        return [f"metadata missing {sorted(missing)}"]
    errors = []
    if isinstance(meta["n_steps"], bool) or not isinstance(meta["n_steps"], int) or meta["n_steps"] < 1:
        errors.append("metadata n_steps must be a positive integer")
    dt = meta["dt"]
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not np.isfinite(dt) or dt <= 0:
        errors.append("metadata dt must be finite and positive")
    try:
        system = get_system(meta["system"])
    except (TypeError, ValueError) as error:
        return errors + [f"metadata system: {error}"]
    for key in ("state_names", "param_names"):
        if meta[key] != getattr(system, key):
            errors.append(f"metadata {key} must be ordered as {getattr(system, key)}")
    return errors


def schema_errors(ds: dict, meta: dict | None = None) -> list[str]:
    """Return shape/dtype errors, including malformed scalar and non-array fields."""
    errors = []
    missing = SCHEMA.keys() - ds.keys()
    if missing:
        return [f"missing {sorted(missing)}"]
    for key in SCHEMA:
        if not isinstance(ds[key], np.ndarray):
            errors.append(f"{key} must be a numpy array")
    for key in ("times", "trajectory_id"):
        if isinstance(ds[key], np.ndarray) and ds[key].ndim != 1:
            errors.append(f"{key} must be one-dimensional")
    if errors:
        return errors
    if meta is not None:
        errors = metadata_errors(meta)
        if errors:
            return errors

    meta = meta or {}
    system = get_system(meta.get("system", "pendulum"))
    dims = {
        "N": len(ds["trajectory_id"]),
        "T1": meta.get("n_steps", len(ds["times"]) - 1) + 1,
        "S": len(system.state_names),
        "P": len(system.param_names),
    }
    for key, (symbols, dtype) in SCHEMA.items():
        wanted = tuple(dims[symbol] for symbol in symbols)
        if ds[key].shape != wanted:
            errors.append(f"{key} shape {ds[key].shape} != {wanted}")
        correct_dtype = ds[key].dtype.kind == "U" if dtype is None else ds[key].dtype == dtype
        if not correct_dtype:
            errors.append(f"{key} dtype {ds[key].dtype} != {dtype or 'str'}")
    return errors


def dataset_errors(ds: dict, meta: dict | None = None) -> list[str]:
    """Return contract violations. Empty split archives are valid."""
    errors = schema_errors(ds, meta)
    if errors:
        return errors

    def require(condition, message):
        if not np.all(condition):
            errors.append(message)

    times = ds["times"]
    require(len(times) >= 2, "times must contain at least two samples")
    if len(times) >= 2:
        dt = (meta or {}).get("dt", times[1] - times[0])
        require(np.isfinite(dt) and dt > 0, "dt must be finite and positive")
        require(
            np.all(np.isfinite(times))
            and np.allclose(times, np.arange(len(times)) * dt, rtol=1e-12, atol=1e-12),
            "times must start at zero and follow the uniform dt grid",
        )

    ids = ds["trajectory_id"]
    require(
        len(np.unique(ids)) == len(ids),
        "trajectory_id must be unique across all splits",
    )
    require(ids >= 0, "trajectory_id must be nonnegative")
    enums = {
        "split": SPLITS,
        "regime": REGIMES,
        "perturbation_type": PERTURBATION_TYPES,
        "severity": ("none",) + SEVERITIES,
        "expected_verdict": VERDICTS,
    }
    for key, choices in enums.items():
        require(np.isin(ds[key], choices), f"{key} must be one of {choices}")
    # Unknown enum values cannot safely be used in the label rules below.
    if any(not np.all(np.isin(ds[key], choices)) for key, choices in enums.items()):
        return errors

    ptype, onset = ds["perturbation_type"], ds["onset_step"]
    none = ptype == "none"
    require(
        np.where(none, onset == -1, (onset >= 0) & (onset < len(times))),
        "onset_step must be -1 for none, otherwise within the sample grid",
    )
    require(ds["severity"][none] == "none", "severity must be none for unperturbed runs")
    for kind, subtypes in SUBTYPES.items():
        require(
            np.isin(ds["perturbation_subtype"][ptype == kind], subtypes),
            f"perturbation_subtype for {kind} must be one of {subtypes}",
        )
    wanted = [expected_verdict(regime, kind) for regime, kind in zip(ds["regime"], ptype)]
    require(
        ds["expected_verdict"] == np.asarray(wanted, dtype=str),
        "expected_verdict does not follow the regime/perturbation rule",
    )
    train_val = np.isin(ds["split"], ("train", "val"))
    require(
        (ds["regime"][train_val] == "gentle") & none[train_val],
        "train/val must contain only gentle, unperturbed trajectories",
    )

    for key in ("true_states", "true_energy", "nominal_params", "actual_params"):
        require(np.isfinite(ds[key]), f"{key} must be finite")
    unshifted = ptype != "param_shift"
    require(
        ds["actual_params"][unshifted] == ds["nominal_params"][unshifted],
        "actual_params must equal nominal_params outside param_shift",
    )

    true, measured, mask = ds["true_states"], ds["measured_states"], ds["measured_mask"]
    require((mask == 0) | (mask == 1), "measured_mask must contain only 0 or 1")
    require(
        np.isnan(measured) == (mask == 0),
        "measured_states must be NaN exactly where measured_mask is 0",
    )
    require(np.isfinite(measured[mask == 1]), "valid measured_states must be finite")
    sensor_fault = ptype == "sensor_fault"
    require(
        measured[~sensor_fault] == true[~sensor_fault],
        "measured_states must equal truth outside sensor_fault",
    )
    require(mask[~sensor_fault] == 1, "measured_mask must be 1 outside sensor_fault")
    before = sensor_fault[:, None] & (np.arange(len(times))[None, :] < onset[:, None])
    require(
        measured[before] == true[before],
        "measured_states must equal truth before sensor onset",
    )
    dropout = sensor_fault & (ds["perturbation_subtype"] == "dropout")
    require(mask[~dropout] == 1, "only dropout may produce missing measured readings")
    system = get_system((meta or {}).get("system", "pendulum"))
    errors.extend(system.dataset_errors(ds))
    return errors


def validate_dataset(ds: dict, meta: dict | None = None) -> None:
    """Raise ValueError with actionable diagnostics before data crosses an interface."""
    errors = dataset_errors(ds, meta)
    if errors:
        raise ValueError("invalid dataset: " + "; ".join(errors))
