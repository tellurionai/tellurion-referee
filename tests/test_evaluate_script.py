"""End-to-end tests for scripts/evaluate.py (T40) on a small generated dataset.

A stand-in "ensemble" (five small-angle pendulums with slightly different g) replaces
the trained model so the test is fast and needs no checkpoint. Like the real model, it
is right on gentle swings and wrong on large ones.
"""
import json
from pathlib import Path

import evaluate as runner
import numpy as np
import pytest

from referee.referee import Referee
from tellurion_sim.dataset import generate, load_config, metadata, save_dataset

ROOT = Path(__file__).resolve().parents[1]

SMALL = {**load_config(ROOT / "configs/pendulum.yaml"), "n_steps": 60,
         "splits": {"train": 40, "val": 10, "cal": 80, "test": 40}, "onset": {"min": 10, "max": 30}}


class SmallAngleEnsemble:
    """forecast(x [B,2], k) -> (mean, std) [B,k,2], like referee.model.Ensemble."""

    def __init__(self, dt, gs=(9.71, 9.76, 9.81, 9.86, 9.91), L=1.0):
        self.dt, self.gs, self.L = dt, gs, L
        self.checkpoint_sha256 = ["stand-in"]

    def forecast(self, x, k):
        x = np.asarray(x, dtype=float)
        runs = []
        for g in self.gs:
            th, om, steps = x[:, 0].copy(), x[:, 1].copy(), []
            for _ in range(k):
                om = om - (g / self.L) * th * self.dt
                th = th + om * self.dt
                steps.append(np.stack([th, om], -1))
            runs.append(np.stack(steps, 1))
        runs = np.stack(runs)
        return runs.mean(0), runs.std(0)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("eval") / "d"
    save_dataset(generate(SMALL, seed=0, workers=1), metadata(SMALL, 0), out)
    return out


def run(dataset, tmp_path, *extra):
    ens = SmallAngleEnsemble(dt=float(SMALL.get("dt", 0.05)))
    args = ["--config", "", "--data", str(dataset), "--results", str(tmp_path / "results"),
            "--steps-out", str(tmp_path / "steps"), "--bootstrap", "50", *extra]
    return runner.main(args, ensemble=ens)


def test_cal_run_writes_every_metric(dataset, tmp_path):
    report = run(dataset, tmp_path)
    saved = json.loads((tmp_path / "results/cal_metrics.json").read_text())
    assert saved["split"] == "cal" and saved["summary"]["n_trajectories"] == 80
    for m in ("referee", "b1_ensemble_uncertainty", "b2_residual", "oracle"):
        d = saved["methods"][m]["detection"]
        assert {"auroc", "false_alarm_rate", "miss_rate", "delay_steps_median", "ci"} <= set(d)
        assert set(d["ci"]) == {"auroc", "miss_rate", "false_alarm_rate", "delay_steps_median"}
    for m in ("referee", "b1_ensemble_uncertainty", "b2_residual", "b3_majority_verdict"):
        diag = saved["methods"][m]["diagnosis"]
        assert len(diag["confusion"]) == 4 and "macro_f1_ci" in diag
    assert (tmp_path / "results/cal_trajectories.csv").read_text().count("\n") == 81
    assert (tmp_path / "steps/cal_steps.npz").exists()
    assert report["dataset"]["sha256"] and report["settings"]["thresholds_fit_on"].startswith("cal")


def test_shared_threshold_caps_false_alarms_on_clean_cal(dataset, tmp_path):
    report = run(dataset, tmp_path)
    for m in ("referee", "b1_ensemble_uncertainty", "b2_residual"):
        assert report["methods"][m]["detection"]["false_alarm_rate"] <= 0.05 + 1e-9
    assert report["referee_fit"]["shared_residual_threshold"] == report["methods"]["referee"]["threshold"]


def test_referee_score_matches_referee_step(dataset):
    """The vectorized detection score is exactly what Referee.step returns as 'score'."""
    cal = runner.load_split(dataset, "cal")
    ens = SmallAngleEnsemble(dt=float(cal["times"][1] - cal["times"][0]))
    mean, _ = runner.one_step_forecasts(ens, cal["measured_states"], cal["measured_mask"])
    scores = runner.referee_scores(cal["measured_states"], cal["measured_mask"], mean)
    ref = Referee()
    checked = 0
    for i in range(0, 80, 9):
        for t in (25, 50):
            if np.isfinite(scores[i, t]):
                out = ref.step(runner.history(cal, i, t, mean))
                assert out["score"] == pytest.approx(scores[i, t], rel=1e-6, abs=1e-9)
                checked += 1
    assert checked > 5


def test_history_never_carries_labels(dataset):
    cal = runner.load_split(dataset, "cal")
    mean = np.zeros_like(cal["measured_states"], dtype=float)
    h = runner.history(cal, 0, 30, mean)
    from referee.referee import BANNED_KEYS
    assert not BANNED_KEYS & set(h)


def test_locked_test_needs_confirmation_and_runs_once(dataset, tmp_path):
    with pytest.raises(SystemExit, match="confirm-locked-test"):
        run(dataset, tmp_path, "--split", "test")
    report = run(dataset, tmp_path, "--split", "test", "--confirm-locked-test")
    assert report["split"] == "test" and report["summary"]["n_trajectories"] == 40
    with pytest.raises(SystemExit, match="already evaluated"):
        run(dataset, tmp_path, "--split", "test", "--confirm-locked-test")
    run(dataset, tmp_path, "--split", "test", "--confirm-locked-test", "--rerun-reason", "unit test")
    assert "unit test" in (tmp_path / "results/RERUNS.md").read_text()
