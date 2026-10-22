"""Run the referee and every baseline on one split and write results/ (T40).

    python scripts/evaluate.py --config configs/base.yaml --split cal     # development, any number of times
    python scripts/evaluate.py --config configs/base.yaml --split test \\
        --confirm-locked-test                                             # ONCE, at Gate 3

Pipeline (plan-referee.md §3.4, §3.5):
1. Load the cal split, plus test when ``--split test``. This script is the only place
   allowed to open the locked test archive (tests/test_no_test_leakage.py).
2. One-step forecasts from the frozen ensemble for every step: the forecast for step t is
   made from the measured state at t - 1. Steps whose previous reading was dropped get no
   forecast (NaN) and therefore no score; NaN scores never flag.
3. Every threshold is fit on clean in-distribution cal runs only (answer key ``trust``),
   with the one shared procedure ``referee.evaluate.fit_threshold`` (at most 5% of clean
   runs flagged). The referee's sensor / physics thresholds come from
   ``Referee.fit_thresholds`` on clean cal windows. The test split never tunes anything.
4. Score every method on the same trajectories with the metrics in ``referee.evaluate``,
   with confidence intervals that resample whole trajectories.

The referee's detection score is its forecast residual (what ``Referee.step`` returns as
``score``). Its detection threshold is replaced by the shared one, so every method gets the
same false-alarm budget. ``Referee.step`` is called where its verdict is graded: at each
trajectory's first sustained flag. Below the threshold ``step`` returns ``trust`` by
construction, so skipping it there changes nothing.

Labels (answer key, onset, severity, true states) are read here only to grade and to pick
clean calibration runs. They are never passed to the referee or the baselines.

Writes:
- results/<split>_metrics.json       every number, its CI and N, thresholds and provenance
- results/<split>_trajectories.csv   one row per trajectory and method
- artifacts/eval/<split>_steps.npz    per-step forecasts and scores (for the replay exporter)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from referee import baselines as B
from referee import evaluate as E
from referee.referee import Referee
from tellurion_sim.dataset import load_dataset

WINDOW = 21               # referee window length in steps (same as the referee tests)
CAL_WINDOW_ENDS = (40, 100, 160)   # clean cal windows used for Referee.fit_thresholds
LOCKED_TEST_RESULTS = "test_metrics.json"
DETECTION_METHODS = ("referee", "b1_ensemble_uncertainty", "b2_residual", "oracle")


# --- inputs ----------------------------------------------------------------------

def load_split(data: Path, split: str) -> dict:
    ds = load_dataset(data, splits=(split,))
    ds["nominal"] = [dict(zip(("g", "L", "m", "b"), map(float, row))) for row in ds["nominal_params"]]
    return ds


def clean_mask(ds: dict) -> np.ndarray:
    """Clean in-distribution runs: answer key ``trust`` (gentle, unperturbed)."""
    return np.asarray(ds["expected_verdict"]) == "trust"


def state_scale(train_states: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """[2] std of valid train states: the normalization every score uses."""
    valid = train_mask.astype(bool).all(-1)
    return train_states[valid].std(axis=0).astype(float)


def one_step_forecasts(ensemble, states: np.ndarray, mask: np.ndarray, chunk: int = 50_000):
    """(mean, std) [N, T, 2]: forecast for step t from the measured state at t - 1; NaN if none."""
    n, t, _ = states.shape
    mean = np.full((n, t, 2), np.nan)
    std = np.full((n, t, 2), np.nan)
    prev = states[:, :-1]
    ok = mask[:, :-1].astype(bool).all(-1) & np.isfinite(prev).all(-1)
    x = prev[ok].astype(np.float32)
    means, stds = [], []
    for start in range(0, len(x), chunk):
        m, s = ensemble.forecast(x[start:start + chunk], 1)
        means.append(np.asarray(m)[:, 0])
        stds.append(np.asarray(s)[:, 0])
    if means:
        mean[:, 1:][ok] = np.concatenate(means)
        std[:, 1:][ok] = np.concatenate(stds)
    return mean, std


def referee_scores(states: np.ndarray, mask: np.ndarray, forecast_mean: np.ndarray) -> np.ndarray:
    """[N, T] the referee's residual (features.forecast_residual), NaN where unmeasured."""
    d = states - forecast_mean
    r = np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2)
    return np.where(mask.astype(bool).all(-1), r, np.nan)


def history(ds: dict, i: int, t: int, forecast_mean: np.ndarray) -> dict:
    """The referee's input for the window ending at step t. No label keys."""
    lo = max(0, t - WINDOW + 1)
    s = ds["measured_states"][i, lo:t + 1]
    return {
        "dt": float(ds["times"][1] - ds["times"][0]),
        "measured_theta": s[:, 0], "measured_omega": s[:, 1],
        "measured_mask": ds["measured_mask"][i, lo:t + 1, 0],
        "nominal_params": ds["nominal"][i],
        "forecast_theta": float(forecast_mean[i, t, 0]),
        "forecast_omega": float(forecast_mean[i, t, 1]),
    }


# --- fitting on cal ----------------------------------------------------------------

def fit_on_cal(cal: dict, cal_fc: dict, scale: np.ndarray) -> dict:
    clean = clean_mask(cal)
    if not clean.any():
        raise ValueError("cal has no clean (trust) trajectories to fit thresholds on")
    scores = method_scores(cal, cal_fc, scale)

    referee = Referee()
    t_max = cal["measured_states"].shape[1] - 1
    ends = [t for t in CAL_WINDOW_ENDS if t <= t_max] or [t_max]
    windows = [history(cal, i, t, cal_fc["mean"]) for i in np.flatnonzero(clean) for t in ends
               if np.isfinite(cal_fc["mean"][i, t]).all()]
    referee.fit_thresholds(windows)
    referee_own = referee.thresholds.residual

    thresholds = {m: E.fit_threshold(scores[m][clean]) for m in DETECTION_METHODS}
    referee.thresholds.residual = thresholds["referee"]   # shared false-alarm budget (§3.4)

    b3 = B.MajorityVerdict()
    b3.fit(cal["expected_verdict"])
    return {"referee": referee, "thresholds": thresholds, "b3": b3,
            "referee_own_residual_threshold": referee_own,
            "referee_check_thresholds": {"sensor_inconsistency": referee.thresholds.sensor_inconsistency,
                                         "fit_error": referee.thresholds.fit_error},
            "cal_clean_windows": len(windows)}


def method_scores(ds: dict, fc: dict, scale: np.ndarray) -> dict:
    states, mask = ds["measured_states"], ds["measured_mask"]
    return {
        "referee": referee_scores(states, mask, fc["mean"]),
        "b1_ensemble_uncertainty": B.spread_scores(fc["std"], scale),
        "b2_residual": B.residual_scores(states, mask, fc["mean"], scale),
        "oracle": B.oracle_scores(ds["true_states"], states, fc["mean_from_true"], scale),
    }


# --- scoring -------------------------------------------------------------------------

def referee_verdicts(ds: dict, referee: Referee, flags: np.ndarray, forecast_mean: np.ndarray):
    """[N, T] step verdicts ('trust' unless flagged) and evidence at each first detection."""
    first = E.detection_steps(flags, None)
    verdicts = np.full(flags.shape, "trust", dtype=object)
    evidence = [""] * len(first)
    for i, t in enumerate(first):
        if t < 0:
            continue
        out = referee.step(history(ds, i, int(t), forecast_mean))
        verdicts[i, t] = out["verdict"]
        evidence[i] = out["evidence"]
    return verdicts.astype(str), evidence


def ci(stat, n: int, samples: int, seed: int) -> dict:
    return E.bootstrap_ci(stat, n, samples=samples, seed=seed)


def detection_block(scores, flags, ds, clean, samples, seed) -> dict:
    onset = np.asarray(ds["onset_step"])
    expected = np.asarray(ds["expected_verdict"])
    out = E.detection_metrics(scores, flags, onset, expected, clean)
    positive = E.is_positive(expected)
    peak = E.sustained_peak(scores)
    detected_at = E.detection_steps(flags, np.where(positive, onset, 0))
    delay = np.where(detected_at >= 0, detected_at - np.maximum(onset, 0), -1)
    any_flag = E.sustained_starts(flags).any(axis=1)
    n = len(expected)

    def auroc(ix):
        return E.auroc(peak[ix], positive[ix])

    def miss(ix):
        p = positive[ix]
        return float((delay[ix][p] < 0).mean()) if p.any() else np.nan

    def far(ix):
        c = clean[ix]
        return float(any_flag[ix][c].mean()) if c.any() else np.nan

    def med_delay(ix):
        d = delay[ix][positive[ix]]
        d = d[d >= 0]
        return float(np.median(d)) if d.size else np.nan

    out["ci"] = {"auroc": ci(auroc, n, samples, seed), "miss_rate": ci(miss, n, samples, seed),
                 "false_alarm_rate": ci(far, n, samples, seed),
                 "delay_steps_median": ci(med_delay, n, samples, seed)}
    sev = np.asarray(ds["severity"])
    out["detected_rate_by_severity"] = {
        str(s): float((delay[positive & (sev == s)] >= 0).mean())
        for s in np.unique(sev[positive])}
    out["_per_trajectory"] = {"peak": peak, "detected_at": detected_at, "delay": delay}
    return out


def evaluate_split(ds: dict, fc: dict, fitted: dict, scale: np.ndarray, samples: int, seed: int):
    clean = clean_mask(ds)
    scores = method_scores(ds, fc, scale)
    flags = {m: E.flag(scores[m], fitted["thresholds"][m]) for m in DETECTION_METHODS}

    error = B.normalized_distance(ds["true_states"], fc["mean"], scale)
    spread = scores["b1_ensemble_uncertainty"]
    cw = E.confidently_wrong(error, spread, clean)

    step_verdicts, evidence = referee_verdicts(ds, fitted["referee"], flags["referee"], fc["mean"])
    expected = np.asarray(ds["expected_verdict"])
    groups = {"severity": ds["severity"], "perturbation_subtype": ds["perturbation_subtype"],
              "regime": ds["regime"]}
    predicted = {
        "referee": E.trajectory_verdicts(step_verdicts, flags["referee"]),
        "b1_ensemble_uncertainty": E.trajectory_verdicts(
            np.where(flags["b1_ensemble_uncertainty"], "model_error", "trust"), flags["b1_ensemble_uncertainty"]),
        "b2_residual": E.trajectory_verdicts(
            np.where(flags["b2_residual"], "model_error", "trust"), flags["b2_residual"]),
        "b3_majority_verdict": fitted["b3"].predict(len(expected)),
    }

    methods = {}
    for m in DETECTION_METHODS:
        methods[m] = {"threshold": fitted["thresholds"][m],
                      "detection": detection_block(scores[m], flags[m], ds, clean, samples, seed),
                      "confidently_wrong_recall": E.recall_on(cw, flags[m])}
    for m, pred in predicted.items():
        block = methods.setdefault(m, {})
        block["diagnosis"] = E.diagnosis_metrics(expected, pred, groups)
        cm_pred = pred

        def f1(ix, p=cm_pred):
            return E.macro_f1(E.confusion_matrix(expected[ix], p[ix]))
        block["diagnosis"]["macro_f1_ci"] = ci(f1, len(expected), samples, seed)
    methods["oracle"]["note"] = "Uses true states: a ceiling for context, not a competitor."

    summary = {"n_trajectories": len(expected), "n_clean": int(clean.sum()),
               "verdict_counts": {v: int((expected == v).sum()) for v in E.VERDICTS},
               "confidently_wrong_steps": int(cw.sum()),
               "confidently_wrong_trajectories": int(cw.any(axis=1).sum())}
    return methods, summary, scores, flags, predicted, evidence


# --- outputs -----------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write_csv(path: Path, ds: dict, methods: dict, predicted: dict, evidence: list[str]) -> None:
    cols = ["trajectory_id", "regime", "perturbation_type", "perturbation_subtype", "severity",
            "onset_step", "expected_verdict"]
    header = cols + [f"{m}_{k}" for m in DETECTION_METHODS for k in ("peak_score", "detected_step", "delay_steps")]
    header += [f"{m}_verdict" for m in predicted] + ["referee_evidence"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(ds["trajectory_id"])):
            row = [ds[c][i].item() if hasattr(ds[c][i], "item") else ds[c][i] for c in cols]
            for m in DETECTION_METHODS:
                pt = methods[m]["detection"]["_per_trajectory"]
                peak = pt["peak"][i]
                row += [None if not np.isfinite(peak) else round(float(peak), 6),
                        int(pt["detected_at"][i]), int(pt["delay"][i])]
            row += [str(predicted[m][i]) for m in predicted] + [evidence[i]]
            w.writerow(row)


def guard_locked_test(results_dir: Path, args) -> None:
    if args.split != "test":
        return
    if not args.confirm_locked_test:
        raise SystemExit("Refusing to touch the locked test split without --confirm-locked-test "
                         "(Gate 3 only, run once).")
    done = results_dir / LOCKED_TEST_RESULTS
    if done.exists():
        if not args.rerun_reason:
            raise SystemExit(f"{done} already exists: the locked test was already evaluated. "
                             "A rerun needs --rerun-reason and is logged in results/RERUNS.md.")
        with open(results_dir / "RERUNS.md", "a", encoding="utf-8") as f:
            f.write(f"- {dt.datetime.now(dt.UTC).isoformat(timespec='seconds')}: "
                    f"commit {git_state()['commit']}: {args.rerun_reason}\n")


def load_ensemble(path: Path):
    from referee.model import load_ensemble as load  # torch is only needed here
    return load(path)


def main(argv=None, ensemble=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--split", choices=("cal", "test"), default="cal")
    ap.add_argument("--data", help="dataset prefix or .npz (default: config data_dir)")
    ap.add_argument("--model", default="artifacts/world_model", help="frozen ensemble directory")
    ap.add_argument("--results", default="results")
    ap.add_argument("--steps-out", default="artifacts/eval")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--confirm-locked-test", action="store_true")
    ap.add_argument("--rerun-reason")
    args = ap.parse_args(argv)

    cfg = {}
    if args.config and Path(args.config).is_file():
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    data = Path(args.data or cfg.get("data_dir", "data/pendulum_v1"))
    data_npz = data if data.suffix == ".npz" else data.with_suffix(".npz")
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    guard_locked_test(results_dir, args)

    if ensemble is None:
        ensemble = load_ensemble(Path(args.model))
    train = load_dataset(data, splits=("train",))
    scale = state_scale(train["measured_states"], train["measured_mask"])

    def forecasts(ds):
        mean, std = one_step_forecasts(ensemble, ds["measured_states"], ds["measured_mask"])
        mean_true, _ = one_step_forecasts(ensemble, ds["true_states"], np.ones_like(ds["measured_mask"]))
        return {"mean": mean, "std": std, "mean_from_true": mean_true}

    cal = load_split(data, "cal")
    cal_fc = forecasts(cal)
    fitted = fit_on_cal(cal, cal_fc, scale)
    ds, fc = (cal, cal_fc) if args.split == "cal" else (load_split(data, "test"), None)
    if fc is None:
        fc = forecasts(ds)

    methods, summary, scores, flags, predicted, evidence = evaluate_split(
        ds, fc, fitted, scale, args.bootstrap, seed)

    meta_path = data_npz.with_suffix(".metadata.json")
    split_file = data_npz if args.split == "cal" else data_npz.with_name(data_npz.stem + ".test.npz")
    report = {
        "split": args.split,
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "command": "python scripts/evaluate.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "git": git_state(),
        "seed": seed,
        "config": cfg,
        "dataset": {"file": str(split_file), "sha256": sha256(split_file),
                    "metadata": json.loads(meta_path.read_text(encoding="utf-8"))},
        "model": {"dir": str(args.model),
                  "checkpoint_sha256": list(getattr(ensemble, "checkpoint_sha256", []))},
        "settings": {"sustain_steps": E.SUSTAIN, "target_false_alarm_rate": E.TARGET_FAR,
                     "confidently_wrong_tau": E.TAU, "confident_percentile": E.CONFIDENT_PCT,
                     "referee_window_steps": WINDOW, "bootstrap_samples": args.bootstrap,
                     "state_scale": scale.tolist(),
                     "thresholds_fit_on": "cal, clean (trust) trajectories only"},
        "referee_fit": {"referee_own_residual_threshold": fitted["referee_own_residual_threshold"],
                        "shared_residual_threshold": fitted["thresholds"]["referee"],
                        "check_thresholds": fitted["referee_check_thresholds"],
                        "cal_clean_windows": fitted["cal_clean_windows"]},
        "summary": summary,
        "methods": methods,
    }
    out_json = results_dir / f"{args.split}_metrics.json"
    out_json.write_text(json.dumps(jsonable(report), indent=2) + "\n", encoding="utf-8")
    write_csv(results_dir / f"{args.split}_trajectories.csv", ds, methods, predicted, evidence)

    steps_dir = Path(args.steps_out)
    steps_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(steps_dir / f"{args.split}_steps.npz", trajectory_id=ds["trajectory_id"],
                        forecast_mean=fc["mean"], forecast_std=fc["std"],
                        **{f"score_{m}": scores[m] for m in DETECTION_METHODS},
                        **{f"flag_{m}": flags[m] for m in DETECTION_METHODS})

    r, b1 = methods["referee"], methods["b1_ensemble_uncertainty"]
    print(f"{args.split}: {summary['n_trajectories']} trajectories ({summary['n_clean']} clean)")
    for name, m in (("referee", r), ("B1 ensemble spread", b1), ("B2 residual", methods["b2_residual"])):
        d = m["detection"]
        print(f"  {name:20s} AUROC {d['auroc']:.3f}  FAR {d['false_alarm_rate']:.3f}  "
              f"miss {d['miss_rate']:.3f}  median delay {d['delay_steps_median']}  "
              f"conf-wrong recall {m['confidently_wrong_recall']:.3f}  macro-F1 {m['diagnosis']['macro_f1']:.3f}")
    print(f"  B3 majority          macro-F1 {methods['b3_majority_verdict']['diagnosis']['macro_f1']:.3f}")
    print(f"wrote {out_json}")
    return report


if __name__ == "__main__":
    main()
