"""Baselines the referee is compared against (plan-referee.md §3.4).

All detection methods take the same per-step inputs as the referee (measured
states + mask, the frozen model's forecasts) and turn them into a score per
step. Every threshold comes from the one procedure in ``evaluate.fit_threshold``,
fit on clean in-distribution ``cal`` trajectories.

- B1 ensemble uncertainty: score = normalized ensemble spread. Detection only;
  a flag means ``model_error``.
- B2 residual threshold: score = normalized one-step forecast residual r_t.
  Detection only; a flag means ``model_error``.
- B3 majority verdict: diagnosis only; always predicts the most common cal label.
- Oracle (context only, never a competitor): uses ``true_states`` to show the
  detection ceiling.

Shapes: states [N, T, 2] as (theta, omega); forecasts aligned to the step they
predict, so ``forecast_mean[:, t]`` is the forecast for step t made at t - 1
(NaN at t = 0); ``scale`` [2] is the model's train state std.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from tellurion_sim.perturbations import VERDICTS

from .evaluate import SUSTAIN, TARGET_FAR, fit_threshold, flag


def _wrapped(diff: np.ndarray) -> np.ndarray:
    """Copy of a state difference with the angle channel wrapped to (-pi, pi]."""
    diff = np.array(diff, dtype=float)
    diff[..., 0] = np.arctan2(np.sin(diff[..., 0]), np.cos(diff[..., 0]))
    return diff


def normalized_distance(a: np.ndarray, b: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """[N, T] L2 of the wrapped state difference divided by ``scale``."""
    return np.linalg.norm(_wrapped(np.asarray(a) - np.asarray(b)) / np.asarray(scale), axis=-1)


def residual_scores(measured: np.ndarray, mask: np.ndarray, forecast_mean: np.ndarray,
                    scale: np.ndarray) -> np.ndarray:
    """B2 score: r_t = |measured_t - f(measured_{t-1})| (normalized). NaN where unmeasured."""
    r = normalized_distance(measured, forecast_mean, scale)
    valid = np.asarray(mask, dtype=bool).all(axis=-1)
    return np.where(valid, r, np.nan)


def spread_scores(forecast_std: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """B1 score: normalized ensemble spread of the forecast for step t."""
    return np.linalg.norm(np.asarray(forecast_std, dtype=float) / np.asarray(scale), axis=-1)


def oracle_scores(true: np.ndarray, measured: np.ndarray, forecast_from_true: np.ndarray,
                  scale: np.ndarray) -> np.ndarray:
    """Ceiling score from ground truth: the larger of the sensor gap |measured - true|
    and the model gap |f(true_{t-1}) - true_t|. Dropped readings (NaN) add no sensor gap."""
    sensor_gap = np.nan_to_num(normalized_distance(measured, true, scale), nan=0.0)
    model_gap = normalized_distance(forecast_from_true, true, scale)
    return np.fmax(sensor_gap, model_gap)


@dataclass
class ThresholdDetector:
    """Detection-only method: flag when a score exceeds a cal-fitted threshold."""

    name: str
    verdict_when_flagged: str = "model_error"
    threshold: float | None = None

    def fit(self, clean_cal_scores: np.ndarray, target_far: float = TARGET_FAR,
            sustain: int = SUSTAIN) -> float:
        self.threshold = fit_threshold(clean_cal_scores, target_far, sustain)
        return self.threshold

    def predict(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(flags [N, T] bool, verdicts [N, T] str)."""
        if self.threshold is None:
            raise ValueError(f"{self.name}: fit on cal before predicting")
        flags = flag(scores, self.threshold)
        return flags, np.where(flags, self.verdict_when_flagged, "trust")


def b1_ensemble_uncertainty() -> ThresholdDetector:
    return ThresholdDetector("B1 ensemble uncertainty")


def b2_residual() -> ThresholdDetector:
    return ThresholdDetector("B2 residual threshold")


def oracle() -> ThresholdDetector:
    # Diagnosis is not part of the oracle; it bounds detection only.
    return ThresholdDetector("oracle (true states)")


@dataclass
class MajorityVerdict:
    """B3: predict the most common answer-key verdict on cal for every trajectory."""

    name: str = "B3 majority verdict"
    verdict: str | None = None

    def fit(self, cal_expected_verdicts: np.ndarray) -> str:
        counts = Counter(np.asarray(cal_expected_verdicts).tolist())
        if not counts:
            raise ValueError("B3 needs cal labels")
        # Ties break in VERDICTS order so the choice is deterministic.
        self.verdict = max(VERDICTS, key=lambda v: (counts.get(v, 0), -VERDICTS.index(v)))
        return self.verdict

    def predict(self, n: int) -> np.ndarray:
        if self.verdict is None:
            raise ValueError("B3: fit on cal before predicting")
        return np.full(n, self.verdict)
