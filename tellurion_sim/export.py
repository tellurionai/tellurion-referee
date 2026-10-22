"""Single-run CSV export for playback and the demo (plan-reality-engine.md §5.3).

predicted_angle_rad, confidence_0to1, pred_x and pred_y are left empty; the
model team fills them in.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .systems.base import System

COLUMNS = [
    "time_s", "true_angle_rad", "measured_angle_rad", "measured_valid", "predicted_angle_rad",
    "confidence_0to1", "true_x", "true_y", "pred_x", "pred_y",
    "perturbation_type", "perturbation_subtype", "severity", "onset_step", "expected_verdict",
]


def write_run_csv(path: str | Path, system: System, times: np.ndarray, run: dict, labels: dict) -> Path:
    """``run`` has true_states, measured_states, measured_mask, actual_params; ``labels``
    has perturbation_type/subtype, severity, onset_step, expected_verdict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    true, meas, mask = run["true_states"], run["measured_states"], run["measured_mask"]
    # Bob position with the length reality used at each step (L changes at onset for length shifts).
    onset = labels["onset_step"]
    L = np.where((onset >= 0) & (np.arange(len(times)) >= onset), run["actual_params"][1], run["nominal_params"][1])
    tx, ty = L * np.sin(true[:, 0]), -L * np.cos(true[:, 0])
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for k, t in enumerate(times):
            valid = int(mask[k, 0])
            w.writerow([
                f"{t:.4f}", f"{true[k, 0]:.8f}", f"{meas[k, 0]:.8f}" if valid else "", valid, "",
                "", f"{tx[k]:.8f}", f"{ty[k]:.8f}", "", "",
                labels["perturbation_type"], labels["perturbation_subtype"], labels["severity"],
                onset, labels["expected_verdict"],
            ])
    return path
