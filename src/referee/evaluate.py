"""Evaluation metrics, fixed before any runs (plan-referee.md §3.5).

Every method (referee, baselines, oracle) is scored the same way. Arrays are
per trajectory and per step: ``scores`` / ``flags`` / ``verdicts`` have shape
[N, T]; labels (``onset_step``, ``expected_verdict``, ...) have shape [N].
Labels are used here only to grade, never as method inputs.

Conventions (decided here so every method is graded identically):
- A *detection* is a sustained flag: ``SUSTAIN`` consecutive flagged steps. Its
  time is the first step of that run.
- A trajectory is *positive* when its answer key is not ``trust`` (clean large /
  rotating runs are positive: the model is out of range from step 0).
- Detection delay counts from ``onset_step`` (0 for positives without an onset)
  to the first detection at or after it.
- AUROC is over trajectories, each scored by its peak sustained step score
  (``sustained_peak``): "per step, aggregated by trajectory".
- Confidence intervals resample whole trajectories (``bootstrap_ci``).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import rankdata

from tellurion_sim.perturbations import VERDICTS

SUSTAIN = 3          # consecutive flagged steps that count as a detection
TAU = 0.25           # normalized forecast error that counts as "wrong"
CONFIDENT_PCT = 80   # confidence percentile on clean data that counts as "confident"
TARGET_FAR = 0.05    # false alarms allowed on clean in-distribution cal runs


# --- sustained flags and the shared threshold procedure -----------------------

def sustained_starts(flags: np.ndarray, sustain: int = SUSTAIN) -> np.ndarray:
    """[N, T] bool: True at t when flags[t : t + sustain] are all True."""
    flags = np.atleast_2d(np.asarray(flags, dtype=bool))
    n, t = flags.shape
    if t < sustain:
        return np.zeros_like(flags)
    windows = np.lib.stride_tricks.sliding_window_view(flags, sustain, axis=1).all(axis=-1)
    return np.concatenate([windows, np.zeros((n, sustain - 1), dtype=bool)], axis=1)


def sustained_peak(scores: np.ndarray, sustain: int = SUSTAIN) -> np.ndarray:
    """[N]: the largest threshold a trajectory would still be flagged above.

    A trajectory has a sustained flag at threshold ``thr`` iff some window of
    ``sustain`` steps has min(score) > thr, so this is max over windows of the
    window minimum. NaN steps (no score, e.g. a dropped reading) never flag.
    """
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    filled = np.where(np.isnan(scores), -np.inf, scores)
    if filled.shape[1] < sustain:
        return np.full(filled.shape[0], -np.inf)
    windows = np.lib.stride_tricks.sliding_window_view(filled, sustain, axis=1).min(axis=-1)
    return windows.max(axis=1)


def fit_threshold(clean_scores: np.ndarray, target_far: float = TARGET_FAR, sustain: int = SUSTAIN) -> float:
    """Smallest threshold with at most ``target_far`` of clean trajectories flagged.

    The one threshold procedure for every method (referee score, B1, B2, oracle):
    fit on clean in-distribution ``cal`` trajectories only. A trajectory is flagged
    when its score exceeds the threshold for ``sustain`` consecutive steps.
    """
    peaks = np.sort(sustained_peak(clean_scores, sustain))[::-1]
    if peaks.size == 0:
        raise ValueError("fit_threshold needs at least one clean trajectory")
    allowed = int(np.floor(target_far * peaks.size))
    return float(peaks[min(allowed, peaks.size - 1)])


def flag(scores: np.ndarray, threshold: float) -> np.ndarray:
    """[N, T] bool per-step flags; NaN scores never flag."""
    scores = np.asarray(scores, dtype=float)
    return np.nan_to_num(scores, nan=-np.inf) > threshold


def detection_steps(flags: np.ndarray, onset_step: np.ndarray | None = None, sustain: int = SUSTAIN) -> np.ndarray:
    """[N] int: first sustained-flag step at or after onset (0 if no onset); -1 if none."""
    starts = sustained_starts(flags, sustain)
    n, t = starts.shape
    first = np.zeros(n, dtype=int) if onset_step is None else np.maximum(np.asarray(onset_step), 0)
    starts = starts & (np.arange(t)[None, :] >= first[:, None])
    return np.where(starts.any(axis=1), starts.argmax(axis=1), -1)


# --- detection ----------------------------------------------------------------

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney U, ties count half). NaN if one class is empty."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def is_positive(expected_verdict: np.ndarray) -> np.ndarray:
    return np.asarray(expected_verdict) != "trust"


def detection_metrics(scores: np.ndarray, flags: np.ndarray, onset_step: np.ndarray,
                      expected_verdict: np.ndarray, clean_in_distribution: np.ndarray,
                      sustain: int = SUSTAIN) -> dict:
    """Detection AUROC, delay, miss rate and false-alarm rates for one method.

    ``clean_in_distribution`` marks the negatives used for the false-alarm rate
    (clean gentle runs, i.e. answer key ``trust``).
    """
    positive = is_positive(expected_verdict)
    clean = np.asarray(clean_in_distribution, dtype=bool)
    onset = np.asarray(onset_step)
    detected_at = detection_steps(flags, np.where(positive, onset, 0), sustain)
    delay = np.where(detected_at >= 0, detected_at - np.maximum(onset, 0), -1)
    pos_delay = delay[positive]
    found = pos_delay >= 0
    any_flag = sustained_starts(flags, sustain).any(axis=1)

    # Flags before onset on perturbed runs are false alarms too (onset > 0 only).
    starts = sustained_starts(flags, sustain)
    before = np.arange(starts.shape[1])[None, :] < onset[:, None]
    has_onset = positive & (onset > 0)
    pre_onset = (starts & before).any(axis=1)

    return {
        "n_positive": int(positive.sum()),
        "n_clean": int(clean.sum()),
        "auroc": auroc(sustained_peak(scores, sustain), positive),
        "detected": int(found.sum()),
        "miss_rate": float(1 - found.mean()) if found.size else float("nan"),
        "delay_steps_median": float(np.median(pos_delay[found])) if found.any() else None,
        "false_alarm_rate": float(any_flag[clean].mean()) if clean.any() else float("nan"),
        "pre_onset_false_alarm_rate": float(pre_onset[has_onset].mean()) if has_onset.any() else float("nan"),
        "delay_steps": pos_delay.tolist(),  # per positive trajectory; -1 = missed
    }


# --- confidently wrong ----------------------------------------------------------

def confidently_wrong(forecast_error: np.ndarray, ensemble_std: np.ndarray, clean: np.ndarray,
                      tau: float = TAU, pct: float = CONFIDENT_PCT) -> np.ndarray:
    """[N, T] bool: the model is wrong (error > tau) while confident.

    Confidence is low ensemble spread. "Confident" means confidence above its
    ``pct``-th percentile on clean trajectories, i.e. spread below the
    (100 - pct)-th percentile of clean spread. Inputs are normalized scalars per step.
    """
    error, std = np.asarray(forecast_error, dtype=float), np.asarray(ensemble_std, dtype=float)
    clean_std = std[np.asarray(clean, dtype=bool)]
    clean_std = clean_std[np.isfinite(clean_std)]
    if clean_std.size == 0:
        raise ValueError("confidently_wrong needs clean trajectories with finite spread")
    cutoff = np.percentile(clean_std, 100 - pct)
    return (np.nan_to_num(error, nan=-np.inf) > tau) & (np.nan_to_num(std, nan=np.inf) < cutoff)


def recall_on(mask: np.ndarray, flags: np.ndarray) -> float:
    """Fraction of ``mask`` steps that are flagged; NaN if there are none."""
    mask = np.asarray(mask, dtype=bool)
    return float(np.asarray(flags, dtype=bool)[mask].mean()) if mask.any() else float("nan")


# --- diagnosis ----------------------------------------------------------------

def trajectory_verdicts(step_verdicts: np.ndarray, flags: np.ndarray, sustain: int = SUSTAIN) -> np.ndarray:
    """[N] verdict per trajectory: the verdict at its first sustained flag, else ``trust``."""
    step_verdicts = np.atleast_2d(np.asarray(step_verdicts))
    first = detection_steps(flags, None, sustain)
    picked = step_verdicts[np.arange(len(first)), np.maximum(first, 0)]
    return np.where(first >= 0, picked, "trust")


def confusion_matrix(true: np.ndarray, pred: np.ndarray, labels=VERDICTS) -> np.ndarray:
    """[K, K] counts; rows = answer key, columns = predicted."""
    index = {v: i for i, v in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(np.asarray(true), np.asarray(pred)):
        cm[index[t], index[p]] += 1
    return cm


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    support = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(support > 0, np.diag(cm) / support, np.nan)


def macro_f1(cm: np.ndarray) -> float:
    """Mean F1 over classes present in the answer key (absent classes are skipped)."""
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(support + predicted > 0, 2 * tp / (support + predicted), 0.0)
    present = support > 0
    return float(f1[present].mean()) if present.any() else float("nan")


def diagnosis_metrics(true: np.ndarray, pred: np.ndarray, groups: dict[str, np.ndarray] | None = None,
                      labels=VERDICTS) -> dict:
    """Confusion matrix, macro-F1 and per-class recall, overall and per group.

    ``groups`` maps a breakdown name (e.g. "severity", "perturbation_subtype") to
    an [N] array of group values.
    """
    true, pred = np.asarray(true), np.asarray(pred)

    def summary(mask):
        cm = confusion_matrix(true[mask], pred[mask], labels)
        recall = per_class_recall(cm)
        return {"n": int(mask.sum()), "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
                "macro_f1": macro_f1(cm), "confusion": cm.tolist(),
                "recall": {v: (None if np.isnan(r) else float(r)) for v, r in zip(labels, recall)}}

    out = {"labels": list(labels), **summary(np.ones(len(true), dtype=bool))}
    for name, values in (groups or {}).items():
        values = np.asarray(values)
        out[f"by_{name}"] = {str(g): summary(values == g) for g in np.unique(values)}
    return out


# --- uncertainty --------------------------------------------------------------

def bootstrap_ci(statistic: Callable[[np.ndarray], float], n: int, samples: int = 1000,
                 seed: int = 0, confidence: float = 0.95) -> dict:
    """Percentile CI resampling whole trajectories.

    ``statistic(indices)`` computes the metric on the trajectories at ``indices``
    (with repeats). Resamples where it is undefined (NaN, e.g. no positives) are
    dropped and counted.
    """
    if n < 1 or samples < 1:
        raise ValueError("bootstrap_ci needs n >= 1 and samples >= 1")
    rng = np.random.default_rng(seed)
    point = float(statistic(np.arange(n)))
    draws = np.array([statistic(rng.integers(0, n, n)) for _ in range(samples)], dtype=float)
    valid = draws[np.isfinite(draws)]
    alpha = (1 - confidence) / 2
    lo, hi = (np.quantile(valid, [alpha, 1 - alpha]) if valid.size else (np.nan, np.nan))
    return {"point": point, "ci95": [float(lo), float(hi)], "n": n,
            "samples": samples, "undefined_samples": int(samples - valid.size), "seed": seed}
