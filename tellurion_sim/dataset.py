"""Dataset generation, splits, and the npz + metadata.json writer (plan-reality-engine.md §5).

The locked ``test`` split is written to its own file, ``<name>.test.npz``, next
to ``<name>.npz`` (train/val/cal). Loading train/val/cal never opens it.

Every trajectory gets its own child RNG from ``SeedSequence(seed)``, so the
output is byte-identical for the same seed and config regardless of how many
worker processes generate it.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from . import integrate
from .perturbations import (
    PERTURBATION_TYPES,
    SEVERITIES,
    SUBTYPES,
    Perturbation,
    apply_sensor_fault,
    expected_verdict,
    force_fn,
    impulse_fn,
    shifted_params,
)
from .systems import get_system
from .systems.pendulum import REGIMES

SPLITS = ("train", "val", "cal", "test")
DEV_SPLITS = ("train", "val", "cal")
LOCKED_SPLIT = "test"


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


@dataclass(frozen=True)
class RunSpec:
    trajectory_id: int
    split: str
    regime: str
    perturbation_type: str
    perturbation_subtype: str
    severity: str
    onset_step: int


def validate_onset(perturbation_type: str, onset_step: int, n_times: int) -> None:
    """-1 is reserved for ``none``; every other perturbation needs 0 <= onset_step < n_times."""
    if perturbation_type == "none":
        if onset_step != -1:
            raise ValueError(f"perturbation 'none' must have onset_step -1, got {onset_step}")
    elif not 0 <= onset_step < n_times:
        raise ValueError(f"{perturbation_type} needs 0 <= onset_step < {n_times}, got {onset_step}")


def simulate_run(cfg: dict, spec: RunSpec, rng: np.random.Generator) -> dict:
    """One labelled trajectory. ``rng`` draws the initial state, then any sensor noise."""
    system = get_system(cfg["system"])
    times = integrate.time_grid(cfg["dt"], cfg["n_steps"])
    validate_onset(spec.perturbation_type, spec.onset_step, len(times))
    pert = Perturbation.from_config(cfg, spec.perturbation_type, spec.perturbation_subtype,
                                    spec.severity, spec.onset_step)
    x0 = system.sample_initial_state(rng, spec.regime)
    nominal = system.nominal_params()
    p_after = shifted_params(nominal, pert, system.param_names)
    t_onset = times[pert.onset_step] if pert.onset_step >= 0 else 0.0

    true, params = integrate.simulate(
        system, x0, times, nominal, p_after,
        force_fn=force_fn(pert, cfg, t_onset), onset_step=pert.onset_step, on_onset=impulse_fn(pert),
    )
    measured, mask = apply_sensor_fault(true, pert, rng)
    energy = np.array([system.energy(true[k], params[k]) for k in range(len(times))])
    return {
        "true_states": true, "measured_states": measured, "measured_mask": mask,
        "true_energy": energy, "nominal_params": nominal, "actual_params": p_after,
        "expected_verdict": expected_verdict(spec.regime, spec.perturbation_type),
    }


def plan_runs(cfg: dict, seed: int) -> list[RunSpec]:
    """Who gets which label, before any physics runs.

    train/val: gentle + none only. cal/test: ``clean_gentle_frac`` extra clean
    gentle runs, the rest cycling through regime x perturbation type x severity
    (subtype drawn uniformly), shuffled.
    """
    rng = np.random.default_rng([seed, 0])
    lo, hi = cfg["onset"]["min"], cfg["onset"]["max"]
    cells = [(r, t, s) for r in REGIMES for t in PERTURBATION_TYPES
             for s in (("none",) if t == "none" else SEVERITIES)]
    specs: list[RunSpec] = []

    def add(split, regime, ptype, severity):
        subtype = str(rng.choice(SUBTYPES[ptype]))
        onset = -1 if ptype == "none" else int(rng.integers(lo, hi + 1))
        specs.append(RunSpec(len(specs), split, regime, ptype, subtype, severity, onset))

    for split in SPLITS:
        n = int(cfg["splits"][split])
        if split in ("train", "val"):
            for _ in range(n):
                add(split, "gentle", "none", "none")
            continue
        n_clean = round(n * cfg["clean_gentle_frac"])
        mix = [("gentle", "none", "none")] * n_clean + [cells[i % len(cells)] for i in range(n - n_clean)]
        for i in rng.permutation(len(mix)):
            add(split, *mix[i])
    return specs


def _simulate_chunk(args):
    cfg, specs, seeds = args
    return [simulate_run(cfg, s, np.random.default_rng(q)) for s, q in zip(specs, seeds)]


def generate(cfg: dict, seed: int, workers: int | None = None) -> dict:
    """Full dataset as a dict of arrays (data contract §5.1)."""
    specs = plan_runs(cfg, seed)
    seeds = np.random.SeedSequence([seed, 1]).spawn(len(specs))
    chunk = 100
    jobs = [(cfg, specs[i:i + chunk], seeds[i:i + chunk]) for i in range(0, len(specs), chunk)]
    workers = workers or os.cpu_count() or 1
    if workers == 1:
        runs = [r for job in jobs for r in _simulate_chunk(job)]
    else:
        with ProcessPoolExecutor(workers) as ex:
            runs = [r for out in ex.map(_simulate_chunk, jobs) for r in out]

    times = integrate.time_grid(cfg["dt"], cfg["n_steps"])
    stack = lambda k, dtype: np.stack([r[k] for r in runs]).astype(dtype)
    return {
        "times": times,
        "true_states": stack("true_states", np.float32),
        "measured_states": stack("measured_states", np.float32),
        "measured_mask": stack("measured_mask", np.uint8),
        "true_energy": stack("true_energy", np.float32),
        "nominal_params": stack("nominal_params", np.float32),
        "actual_params": stack("actual_params", np.float32),
        "trajectory_id": np.array([s.trajectory_id for s in specs], dtype=np.int64),
        "split": np.array([s.split for s in specs]),
        "regime": np.array([s.regime for s in specs]),
        "perturbation_type": np.array([s.perturbation_type for s in specs]),
        "perturbation_subtype": np.array([s.perturbation_subtype for s in specs]),
        "severity": np.array([s.severity for s in specs]),
        "onset_step": np.array([s.onset_step for s in specs], dtype=np.int32),
        "expected_verdict": np.array([r["expected_verdict"] for r in runs]),
    }


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=Path(__file__).parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def metadata(cfg: dict, seed: int) -> dict:
    system = get_system(cfg["system"])
    return {
        "system": system.name,
        "state_names": system.state_names, "state_units": system.state_units,
        "param_names": system.param_names, "param_units": system.param_units,
        "nominal_params": system.nominal,
        "dt": cfg["dt"], "n_steps": cfg["n_steps"], "horizon_s": cfg["dt"] * cfg["n_steps"],
        "solver": {**integrate.SOLVER, "restart": "every grid step"},
        "seed": seed, "git_commit": _git_commit(),
        "severity_table": {t: cfg[t] for t in ("param_shift", "sensor_fault", "unseen_force")},
        "magnet_phi_m": cfg["magnet_phi_m"], "periodic_drive_w": cfg["periodic_drive_w"],
        "onset": cfg["onset"],
        "split_sizes": cfg["splits"], "clean_gentle_frac": cfg["clean_gentle_frac"],
        "split_rules": "train/val: gentle + none only. cal/test: clean gentle share plus "
                       "regime x type x severity cycle. Split by trajectory_id. test is locked.",
        "sensor_model": "theta and omega both measured; sensor faults corrupt theta only",
        "verdict_rule": "none -> trust in gentle, model_error otherwise; param_shift -> model_error; "
                        "sensor_fault -> instrument_fault; unseen_force -> discovery",
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
    }


def _paths(out: str | Path) -> tuple[Path, Path, Path]:
    """(<name>.npz, <name>.test.npz, <name>.metadata.json) for a prefix or a .npz path."""
    out = Path(out)
    name = out.name.removesuffix(".npz").removesuffix(".test")
    return (out.with_name(name + ".npz"), out.with_name(name + ".test.npz"),
            out.with_name(name + ".metadata.json"))


def save_dataset(ds: dict, meta: dict, out: str | Path) -> tuple[Path, Path, Path]:
    """Write ``<out>.npz`` (train/val/cal), ``<out>.test.npz`` (locked test) and ``<out>.metadata.json``."""
    dev, test, js = _paths(out)
    dev.parent.mkdir(parents=True, exist_ok=True)
    is_test = ds["split"] == LOCKED_SPLIT
    for path, keep in ((dev, ~is_test), (test, is_test)):
        np.savez_compressed(path, **{k: (v if k == "times" else v[keep]) for k, v in ds.items()})
    js.write_text(json.dumps({**meta, "files": {"dev": dev.name, "test": test.name}}, indent=2))
    return dev, test, js


def load_dataset(path: str | Path, splits: tuple[str, ...] = DEV_SPLITS) -> dict:
    """Load a dataset (``<name>`` prefix or ``<name>.npz``) as a dict of arrays.

    Only trajectories in ``splits`` are returned. The locked test file is
    opened only when ``"test"`` is requested, which only the final
    evaluation should ever do.
    """
    unknown = set(splits) - set(SPLITS)
    if unknown:
        raise ValueError(f"unknown splits {sorted(unknown)}; known: {SPLITS}")
    dev, test, _ = _paths(path)
    parts = []
    for file, wanted in ((dev, set(splits) & set(DEV_SPLITS)), (test, set(splits) & {LOCKED_SPLIT})):
        if not wanted:
            continue
        with np.load(file, allow_pickle=False) as z:
            part = {k: z[k] for k in z.files}
        keep = np.isin(part["split"], list(wanted))
        parts.append({k: (v if k == "times" else v[keep]) for k, v in part.items()})
    if not parts:
        raise ValueError("no splits requested")
    return {k: (parts[0][k] if k == "times" else np.concatenate([p[k] for p in parts])) for k in parts[0]}
