"""Metrics (#26) and baselines (#27) on synthetic cases with known answers."""
from itertools import product

import numpy as np
import pytest

from referee import baselines
from referee.evaluate import (
    auroc,
    bootstrap_ci,
    confidently_wrong,
    confusion_matrix,
    detection_metrics,
    detection_steps,
    diagnosis_metrics,
    fit_threshold,
    flag,
    macro_f1,
    per_class_recall,
    recall_on,
    sustained_peak,
    sustained_starts,
    trajectory_verdicts,
)

F, T = False, True


# --- sustained flags + threshold procedure -------------------------------------

def test_sustained_starts_needs_three_in_a_row():
    flags = np.array([[F, T, T, F, T, T, T, T]])
    assert sustained_starts(flags).nonzero()[1].tolist() == [4, 5]


def test_detection_step_respects_onset():
    flags = np.array([[T, T, T, F, F, F, F, T, T, T, F, F]])
    assert detection_steps(flags).tolist() == [0]
    assert detection_steps(flags, np.array([3])).tolist() == [7]
    assert detection_steps(flags, np.array([9])).tolist() == [-1]  # only 2 flagged steps left
    assert detection_steps(flags, np.array([-1])).tolist() == [0]  # no onset counts from 0


def test_sustained_peak_is_the_flag_boundary():
    scores = np.array([[0.1, 0.9, 0.8, 0.7, 0.2], [0.5, 0.5, np.nan, 0.9, 0.9]])
    peak = sustained_peak(scores)
    assert peak.tolist() == [0.7, -np.inf]  # NaN breaks the second run
    for i, p in enumerate(peak):
        if np.isfinite(p):
            assert sustained_starts(flag(scores[i:i + 1], p - 1e-9)).any()
            assert not sustained_starts(flag(scores[i:i + 1], p)).any()


def test_fit_threshold_hits_target_false_alarm_rate():
    rng = np.random.default_rng(0)
    clean = rng.random((100, 50))
    thr = fit_threshold(clean, target_far=0.05)
    far = sustained_starts(flag(clean, thr)).any(axis=1).mean()
    assert far <= 0.05
    # Smallest such threshold: any lower peak value would exceed the target.
    peaks = np.sort(sustained_peak(clean))
    lower = peaks[peaks < thr].max()
    assert sustained_starts(flag(clean, lower)).any(axis=1).mean() > 0.05
    assert fit_threshold(clean, target_far=0.0) == pytest.approx(peaks.max())


# --- detection ------------------------------------------------------------------

def test_auroc_known_values_and_ties():
    assert auroc([0.1, 0.2, 0.8, 0.9], [F, F, T, T]) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [F, F, T, T]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [F, F, T, T]) == 0.5
    assert np.isnan(auroc([0.1, 0.2], [T, T]))


def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 5, 60).astype(float)  # many ties
    y = rng.random(60) < 0.4
    pairs = [(a > b) + 0.5 * (a == b) for a, b in product(s[y], s[~y])]
    assert auroc(s, y) == pytest.approx(np.mean(pairs))


def test_detection_metrics_known_case():
    # 4 trajectories x 10 steps: clean, clean (false alarm), fault onset 3 (caught at 5), fault onset 4 (missed)
    flags = np.zeros((4, 10), dtype=bool)
    flags[1, 2:5] = T
    flags[2, 5:8] = T
    flags[3, 6:8] = T  # only 2 steps: not sustained
    scores = flags.astype(float)
    m = detection_metrics(scores, flags, onset_step=np.array([-1, -1, 3, 4]),
                          expected_verdict=np.array(["trust", "trust", "instrument_fault", "discovery"]),
                          clean_in_distribution=np.array([T, T, F, F]))
    assert m["delay_steps"] == [2, -1]
    assert m["detected"] == 1 and m["miss_rate"] == 0.5
    assert m["delay_steps_median"] == 2.0
    assert m["false_alarm_rate"] == 0.5
    assert m["pre_onset_false_alarm_rate"] == 0.0
    assert m["auroc"] == pytest.approx(0.5)  # one positive peak 1, the other 0; negatives 0 and 1


def test_clean_out_of_range_run_counts_from_step_zero():
    flags = np.zeros((1, 10), dtype=bool)
    flags[0, 4:7] = T
    m = detection_metrics(flags.astype(float), flags, np.array([-1]), np.array(["model_error"]), np.array([F]))
    assert m["delay_steps"] == [4]


def test_pre_onset_flags_are_false_alarms():
    flags = np.zeros((1, 12), dtype=bool)
    flags[0, 1:4] = T
    m = detection_metrics(flags.astype(float), flags, np.array([8]), np.array(["discovery"]), np.array([F]))
    assert m["pre_onset_false_alarm_rate"] == 1.0
    assert m["delay_steps"] == [-1]  # the early flag isn't a detection of the fault


# --- confidently wrong ------------------------------------------------------------

def test_confidently_wrong_needs_error_and_low_spread():
    std = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.1, 0.9, 0.9, 0.9]])
    err = np.array([[0.0] * 5, [0.5, 0.1, 0.5, 0.5, np.nan]])
    clean = np.array([T, F])
    # 20th percentile of clean spread [0.1..0.5] is 0.18: only spread 0.1 is "confident".
    cw = confidently_wrong(err, std, clean)
    assert cw.tolist() == [[F] * 5, [T, F, F, F, F]]
    flags = np.array([[F] * 5, [T, F, F, F, F]])
    assert recall_on(cw, flags) == 1.0
    assert np.isnan(recall_on(np.zeros_like(cw), flags))


# --- diagnosis --------------------------------------------------------------------

def test_trajectory_verdict_is_verdict_at_first_sustained_flag():
    v = np.array([["trust", "discovery", "discovery", "discovery", "model_error"],
                  ["trust"] * 5])
    flags = v != "trust"
    assert trajectory_verdicts(v, flags).tolist() == ["discovery", "trust"]


def test_confusion_and_macro_f1_by_hand():
    true = ["trust", "trust", "model_error", "instrument_fault", "discovery", "discovery"]
    pred = ["trust", "model_error", "model_error", "discovery", "discovery", "discovery"]
    cm = confusion_matrix(true, pred)
    assert cm.tolist() == [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]
    assert per_class_recall(cm).tolist() == [0.5, 1.0, 0.0, 1.0]
    # F1: trust 2/3, model_error 2/3, instrument_fault 0, discovery 0.8
    assert macro_f1(cm) == pytest.approx((2 / 3 + 2 / 3 + 0 + 0.8) / 4)


def test_diagnosis_breakdown_by_group():
    true = np.array(["discovery", "discovery", "instrument_fault"])
    pred = np.array(["discovery", "trust", "instrument_fault"])
    out = diagnosis_metrics(true, pred, {"severity": np.array(["low", "high", "high"])})
    assert out["n"] == 3 and out["accuracy"] == pytest.approx(2 / 3)
    assert out["by_severity"]["low"]["accuracy"] == 1.0
    assert out["by_severity"]["high"]["accuracy"] == 0.5
    assert out["recall"]["trust"] is None  # class absent from the answer key


# --- bootstrap ----------------------------------------------------------------------

def test_bootstrap_resamples_trajectories_deterministically():
    values = np.array([0.0] * 9 + [10.0])
    stat = lambda idx: values[idx].mean()  # noqa: E731
    a, b = bootstrap_ci(stat, 10, samples=500, seed=3), bootstrap_ci(stat, 10, samples=500, seed=3)
    assert a == b
    assert a["point"] == 1.0
    lo, hi = a["ci95"]
    assert lo <= 1.0 <= hi and lo == 0.0  # a resample without the outlier happens often


def test_bootstrap_drops_undefined_resamples():
    labels = np.array([T] + [F] * 9)
    stat = lambda idx: auroc(np.arange(10.0)[idx], labels[idx])  # noqa: E731
    out = bootstrap_ci(stat, 10, samples=200, seed=0)
    assert out["undefined_samples"] > 0  # resamples with no positive trajectory
    assert out["undefined_samples"] < 200


# --- baselines (#27) ----------------------------------------------------------------

SCALE = np.array([0.5, 2.0])


def test_residual_wraps_angle_and_respects_mask():
    measured = np.array([[[np.pi - 0.01, 0.0], [0.3, 0.0]]])
    forecast = np.array([[[-np.pi + 0.01, 0.0], [0.3, 0.0]]])
    mask = np.ones_like(measured, dtype=np.uint8)
    r = baselines.residual_scores(measured, mask, forecast, SCALE)
    assert r[0, 0] == pytest.approx(0.02 / 0.5)
    mask[0, 1, 0] = 0
    assert np.isnan(baselines.residual_scores(measured, mask, forecast, SCALE)[0, 1])


def test_spread_score_is_normalized_norm():
    std = np.array([[[0.3, 0.8]]])
    assert baselines.spread_scores(std, SCALE)[0, 0] == pytest.approx(np.hypot(0.6, 0.4))


def test_oracle_sees_sensor_and_model_gaps():
    true = np.zeros((1, 3, 2))
    measured = true.copy()
    measured[0, 1, 0] = 0.25          # sensor bias at step 1
    forecast = true.copy()
    forecast[0, 2, 1] = 1.0           # model wrong at step 2
    s = baselines.oracle_scores(true, measured, forecast, SCALE)
    assert s[0].tolist() == pytest.approx([0.0, 0.5, 0.5])


def test_detectors_share_the_threshold_procedure():
    rng = np.random.default_rng(2)
    clean = rng.random((40, 30))
    for make in (baselines.b1_ensemble_uncertainty, baselines.b2_residual, baselines.oracle):
        det = make()
        assert det.fit(clean) == fit_threshold(clean)
        flags, verdicts = det.predict(clean)
        assert sustained_starts(flags).any(axis=1).mean() <= 0.05
        assert set(np.unique(verdicts)) <= {"trust", "model_error"}
    with pytest.raises(ValueError, match="fit"):
        baselines.b2_residual().predict(clean)


def test_b1_cannot_tell_causes_apart():
    det = baselines.b1_ensemble_uncertainty()
    det.threshold = 0.5
    _, verdicts = det.predict(np.array([[0.9, 0.9, 0.9]]))
    assert set(verdicts.ravel()) == {"model_error"}


def test_b3_majority_with_deterministic_ties():
    b3 = baselines.MajorityVerdict()
    assert b3.fit(np.array(["discovery", "model_error", "model_error"])) == "model_error"
    assert b3.predict(3).tolist() == ["model_error"] * 3
    assert baselines.MajorityVerdict().fit(np.array(["discovery", "instrument_fault"])) == "instrument_fault"


def test_end_to_end_synthetic_referee_beats_b1_on_confident_errors():
    """A model that is confidently wrong after onset: B1 (spread) misses it, B2 (residual) catches it."""
    n, t = 20, 40
    onset = np.where(np.arange(n) < 10, -1, 15)
    verdict = np.where(np.arange(n) < 10, "trust", "discovery")
    spread = np.full((n, t), 0.05)                     # always confident
    residual = np.full((n, t), 0.05)
    residual[10:, 15:] = 1.0                           # wrong after onset
    clean = verdict == "trust"

    b1, b2 = baselines.b1_ensemble_uncertainty(), baselines.b2_residual()
    b1.fit(spread[clean]), b2.fit(residual[clean])
    f1, _ = b1.predict(spread)
    f2, _ = b2.predict(residual)
    m1 = detection_metrics(spread, f1, onset, verdict, clean)
    m2 = detection_metrics(residual, f2, onset, verdict, clean)
    assert m1["miss_rate"] == 1.0 and m2["miss_rate"] == 0.0
    assert m2["delay_steps_median"] == 0.0 and m2["auroc"] == 1.0
    cw = confidently_wrong(residual, spread + np.linspace(1e-3, 0, t), clean)  # most confident late
    assert recall_on(cw, f2) == 1.0 and recall_on(cw, f1) == 0.0
