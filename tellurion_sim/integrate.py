"""solve_ivp wrapper on a fixed output grid, split at onset.

Integration restarts at every grid point (one solve_ivp call per dt). That
makes the trajectory up to step k independent of anything that happens after
step k, so "before onset the perturbed run equals the clean run" holds
bit-for-bit, and discontinuities (onset, impulse) always land on a restart
instead of being integrated across.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .systems.base import ForceFn, System

SOLVER = {"method": "DOP853", "rtol": 1e-9, "atol": 1e-11}


def time_grid(dt: float, n_steps: int) -> np.ndarray:
    """[n_steps + 1] sample times [s]."""
    return np.arange(n_steps + 1, dtype=np.float64) * dt


def simulate(
    system: System,
    x0: np.ndarray,
    times: np.ndarray,
    p_before: np.ndarray,
    p_after: np.ndarray | None = None,
    force_fn: ForceFn | None = None,
    onset_step: int = -1,
    on_onset: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate ``system`` on ``times``.

    Before ``onset_step`` the run uses ``p_before`` and no external force. From
    ``onset_step`` on it uses ``p_after`` and ``force_fn``; ``on_onset(x, p)``
    may change the state at the onset instant (an impulse). ``onset_step = -1``
    means no perturbation.

    Returns (states [T+1, S] float64, params in effect per step [T+1, P]).
    """
    p_after = p_before if p_after is None else p_after
    n = len(times)
    states = np.empty((n, len(x0)), dtype=np.float64)
    params = np.empty((n, len(p_before)), dtype=np.float64)
    x = np.asarray(x0, dtype=np.float64).copy()

    for k in range(n):
        active = 0 <= onset_step <= k
        p = p_after if active else p_before
        if k == onset_step and on_onset is not None:
            x = on_onset(x, p)
        states[k], params[k] = x, p
        if k == n - 1:
            break
        f = force_fn if active else None
        sol = solve_ivp(
            system.rhs, (times[k], times[k + 1]), x, args=(p, f),
            method=SOLVER["method"], rtol=SOLVER["rtol"], atol=SOLVER["atol"],
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed at step {k}: {sol.message}")
        x = sol.y[:, -1]
    return states, params
