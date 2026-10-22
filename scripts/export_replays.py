"""Export real evaluation results as demo replays (T41; re-run after Gate 3 for T48).

    python scripts/export_replays.py --split cal     # from results of `make eval-cal`
    python scripts/export_replays.py --split <split>  # the locked-test results, after Gate 3

Reads only what scripts/evaluate.py wrote: artifacts/eval/<split>_steps.npz and
results/<split>_metrics.json. It never opens the dataset, so it never touches the locked
test archive itself. Writes demo/public/replays/<id>.json + index.json in the format of
demo/REPLAY_FORMAT.md, with is_demo_data = false, and removes replays it did not write.

Honesty rules (plan-demo.md):
- Every scenario is picked by a fixed, stated rule, never by whether the referee got it
  right: among runs of one kind, the one with the median referee detection delay (clean
  runs: the median peak referee score; ties: lowest trajectory id). Wrong verdicts stay
  in; the demo shows the answer key beside them.
- At least one honest miss: the faulty run the referee never flagged, picked by the
  same rule among missed runs, unless a scenario above is already a miss.

Meters (0 to 1, documented in index.json):
- AI confidence = 2 ** (-spread / B1 threshold): 100% when the five members agree
  exactly, 50% at the spread where the B1 baseline would raise its alarm.
- Referee = 1 - 2 ** (-score / referee threshold): 50% exactly at the referee's alarm line.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo/public/replays"

# id, title, (regime, perturbation_type, subtype or None), description
SCENARIOS = [
    ("normal-swing", "Normal swing", ("gentle", "none", None),
     "A gentle swing, like the ones the AI was trained on."),
    ("big-swing", "Big swing", ("large", "none", None),
     "Same pendulum, pushed much higher than anything the AI saw in training."),
    ("longer-string", "Longer string", ("gentle", "param_shift", "length"),
     "At the switch, the string gets longer. The AI still assumes the old length."),
    ("broken-sensor", "Broken sensor", ("gentle", "sensor_fault", "stuck"),
     "At the switch, the angle sensor freezes. The pendulum keeps swinging."),
    ("drifting-sensor", "Drifting sensor", ("gentle", "sensor_fault", "drift"),
     "At the switch, the angle sensor slowly drifts away from the true angle."),
    ("hidden-magnet", "Hidden magnet", ("gentle", "unseen_force", "magnet"),
     "At the switch, a hidden magnet starts pulling the bob sideways."),
    ("sudden-kick", "Sudden kick", ("gentle", "unseen_force", "impulse"),
     "At the switch, something knocks the pendulum."),
]


def r4(a) -> list:
    return [None if (x is None or not np.isfinite(x)) else round(float(x), 4) for x in np.asarray(a, dtype=float)]


def median_score_pick(s: dict, idx: np.ndarray) -> int:
    """Index of the run with the median peak referee score: a typical run (used for clean runs)."""
    peak = np.nanmax(np.where(np.isfinite(s["score_referee"][idx]), s["score_referee"][idx], -np.inf), axis=1)
    order = np.lexsort((s["trajectory_id"][idx], peak))
    return int(idx[order[(len(order) - 1) // 2]])


def median_delay_pick(s: dict, idx: np.ndarray) -> int:
    """Index (into the split) of the run with the median referee detection delay; missed runs last."""
    onset = np.maximum(s["onset_step"][idx], 0)
    det = s["referee_detected_at"][idx]
    delay = np.where(det >= 0, det - onset, np.iinfo(np.int64).max)
    order = np.lexsort((s["trajectory_id"][idx], delay))
    detected = order[delay[order] != np.iinfo(np.int64).max]
    pool = detected if detected.size else order
    return int(idx[pool[(len(pool) - 1) // 2]])


def candidates(s: dict, regime: str, ptype: str, subtype: str | None) -> tuple[np.ndarray, str]:
    base = (s["regime"] == regime) & (s["perturbation_type"] == ptype)
    if subtype is not None:
        base &= s["perturbation_subtype"] == subtype
    kind = f"{regime}, {subtype or 'clean'}" + ("" if ptype == "none" else ", all severities")
    return np.flatnonzero(base), kind


def replay(s: dict, i: int, sid: str, title: str, description: str) -> dict:
    t = s["times"]
    spread = np.linalg.norm(s["forecast_std"][i] / s["state_scale"], axis=-1)
    confidence = np.power(2.0, -spread / s["threshold_b1"])
    score = s["score_referee"][i]
    referee = 1.0 - np.power(2.0, -score / s["threshold_referee"])
    first = int(s["referee_first_flag"][i])
    verdict = str(s["referee_verdict"][i])
    flagged = first >= 0 and verdict != "trust"
    verdicts = ["trust" if (not flagged or k < first) else verdict for k in range(len(t))]
    evidence = str(s["referee_evidence"][i]) or None
    return {
        "id": sid, "title": title, "description": description,
        "is_demo_data": False,
        "system": "pendulum", "L": float(s["L"][i]), "dt": float(t[1] - t[0]),
        "onset_step": int(s["onset_step"][i]),
        "perturbation": {"type": str(s["perturbation_type"][i]), "subtype": str(s["perturbation_subtype"][i]),
                         "severity": str(s["severity"][i])},
        "expected_verdict": str(s["expected_verdict"][i]),
        "trajectory_id": int(s["trajectory_id"][i]),
        "frames": {
            "t": r4(t),
            "true_theta": r4(s["true_theta"][i]),
            "measured_theta": r4(s["measured_theta"][i]),
            "pred_theta": r4(s["forecast_mean"][i, :, 0]),
            "ai_confidence": r4(confidence),
            "referee_score": r4(referee),
            "verdict": verdicts,
            "evidence": [None if v == "trust" else evidence for v in verdicts],
        },
    }


def load_steps(path: Path, metrics: dict) -> dict:
    with np.load(path, allow_pickle=False) as z:
        s = {k: z[k] for k in z.files}
    s = {k: (v.astype(str) if v.dtype.kind == "U" else v) for k, v in s.items()}
    s["threshold_b1"] = float(s["threshold_b1_ensemble_uncertainty"])
    s["threshold_referee"] = float(s["threshold_referee"])
    s["state_scale"] = np.asarray(metrics["settings"]["state_scale"], dtype=float)
    return s


def export(split: str, steps: Path, metrics_path: Path, out: Path) -> dict:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    s = load_steps(steps, metrics)
    if str(s["split"]) != split or metrics["split"] != split:
        raise SystemExit(f"{steps} / {metrics_path} are not {split} results")
    out.mkdir(parents=True, exist_ok=True)

    index, written, used = [], set(), set()
    for sid, title, (regime, ptype, subtype), description in SCENARIOS:
        idx, kind = candidates(s, regime, ptype, subtype)
        if not idx.size:
            print(f"skip {sid}: no {kind} runs in {split}")
            continue
        typical = (s["expected_verdict"][idx] == "trust").all()
        i = median_score_pick(s, idx) if typical else median_delay_pick(s, idx)
        used.add(i)
        missed = ptype != "none" and int(s["referee_detected_at"][i]) < 0
        what = "peak referee score" if typical else "referee detection delay"
        runs = "run" if idx.size == 1 else "runs"
        rule = f"median {what} among {idx.size} {split} {runs} ({kind}); ties: lowest id"
        index.append(write(out, s, i, sid, title + (" (missed)" if missed else ""), description,
                           rule, missed, written))

    faulty = s["expected_verdict"] != "trust"
    missed_runs = np.flatnonzero(faulty & (s["referee_detected_at"] < 0))
    if not any(e["missed"] for e in index) and missed_runs.size:
        pool = np.array([j for j in missed_runs if j not in used]) if len(used) else missed_runs
        if pool.size:
            j = int(pool[np.lexsort((s["trajectory_id"][pool], s["onset_step"][pool]))][(len(pool) - 1) // 2])
            sub = str(s["perturbation_subtype"][j])
            index.append(write(out, s, j, "honest-miss", f"{sub.replace('_', ' ').capitalize()} (missed)",
                               "A fault the referee never flagged. Shown on purpose.",
                               f"median onset among {pool.size} faulty {split} runs the referee never flagged",
                               True, written))

    for stale in out.glob("*.json"):
        if stale.name != "index.json" and stale.name not in written:
            stale.unlink()
    doc = {
        "generated_from": (f"results/{split}_metrics.json, commit {str(s['git_commit'])[:7] or 'unknown'}, "
                           f"model {str(s['checkpoint_sha256'][0])[:8] if s['checkpoint_sha256'].size else 'unknown'}"),
        "is_demo_data": False,
        "split": split,
        "meters": {"ai_confidence": "2^(-ensemble spread / B1 alarm threshold): 50% at B1's alarm line",
                   "referee_score": "1 - 2^(-referee score / referee alarm threshold): 50% at its alarm line"},
        "scenarios": index,
    }
    (out / "index.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


def write(out: Path, s: dict, i: int, sid: str, title: str, description: str, rule: str,
          missed: bool, written: set) -> dict:
    data = replay(s, i, sid, title, description)
    (out / f"{sid}.json").write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
    written.add(f"{sid}.json")
    got = data["frames"]["verdict"][-1]
    print(f"{sid:16s} id {data['trajectory_id']:5d}  answer {data['expected_verdict']:16s} referee {got}")
    return {"id": sid, "title": title, "file": f"{sid}.json", "expected_verdict": data["expected_verdict"],
            "selection_rule": rule, "missed": missed}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=("cal", "test"), default="cal")
    ap.add_argument("--steps", help="default: artifacts/eval/<split>_steps.npz")
    ap.add_argument("--metrics", help="default: results/<split>_metrics.json")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    steps = Path(a.steps or f"artifacts/eval/{a.split}_steps.npz")
    metrics = Path(a.metrics or f"results/{a.split}_metrics.json")
    for p in (steps, metrics):
        if not p.is_file():
            sys.exit(f"missing {p}: run `make eval-{a.split}` first")
    doc = export(a.split, steps, metrics, Path(a.out))
    print(f"wrote {len(doc['scenarios'])} replays + index.json to {a.out}")
    return doc


if __name__ == "__main__":
    main()
