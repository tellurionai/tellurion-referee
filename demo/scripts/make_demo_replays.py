"""DEMO_ONLY replays for building the UI before the real pipeline exists.

Physics ("reality", sensor) comes from tellurion_sim. The "AI" and "referee"
are crude stand-ins, NOT the team's model or referee:

- AI prediction: a small-angle (linearised) pendulum, which is roughly what a
  model trained only on gentle swings learns, rolled 0.5 s ahead from the
  sensor reading. Confidence is high and barely reacts, by construction.
- Referee score: smoothed gap between the sensor and that prediction. When it
  stays above 0.5 the stand-in "flags" with the run's answer-key verdict.

Every file is marked is_demo_data and the UI watermarks it. Real replays come
from scripts/export_replays.py (T41) and overwrite these.

    uv run --with-requirements requirements.txt python demo/scripts/make_demo_replays.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tellurion_sim import integrate  # noqa: E402
from tellurion_sim.dataset import RunSpec, load_config, simulate_run  # noqa: E402
from tellurion_sim.perturbations import expected_verdict  # noqa: E402

OUT = ROOT / "demo/public/replays"
SEED, ONSET = 3, 80
HORIZON = 10          # steps the stand-in AI predicts ahead (0.5 s)
FLAG_AT, FLAG_HOLD = 0.5, 3

EVIDENCE = {
    "model_error": "Sensors agree with each other and plain pendulum physics fits; the AI's prediction doesn't.",
    "instrument_fault": "The angle reading contradicts the speed reading. The physics is fine; the sensor isn't.",
    "discovery": "Sensors agree, but no pendulum physics fit explains the motion. Something extra is acting on it.",
}

# id, title, regime, type, subtype, severity, blurb
SCENARIOS = [
    ("normal-swing", "Normal swing", "gentle", "none", "none", "none",
     "A gentle swing, like the ones the AI was trained on."),
    ("big-swing", "Big swing", "large", "none", "none", "none",
     "Same pendulum, pushed much higher than anything the AI saw in training."),
    ("longer-string", "Longer string", "gentle", "param_shift", "length", "high",
     "At the switch, the string gets 30% longer. The AI still assumes the old length."),
    ("broken-sensor", "Broken sensor", "gentle", "sensor_fault", "stuck", "high",
     "At the switch, the angle sensor freezes. The pendulum keeps swinging."),
    ("hidden-magnet", "Hidden magnet", "gentle", "unseen_force", "magnet", "med",
     "At the switch, a hidden magnet starts pulling the bob sideways."),
    ("sudden-kick", "Sudden kick", "gentle", "unseen_force", "impulse", "high",
     "At the switch, something knocks the pendulum."),
    ("flaky-sensor", "Flaky sensor", "gentle", "sensor_fault", "dropout", "low",
     "At the switch, the angle sensor starts dropping about 1 in 20 readings."),
]


def ai_forecast(theta: np.ndarray, omega: np.ndarray, valid: np.ndarray, dt: float, g: float, L: float) -> np.ndarray:
    """Prediction for step k, made at step k - HORIZON from the last valid reading."""
    w = math.sqrt(g / L)
    n = len(theta)
    pred = np.full(n, np.nan)
    last = None
    for k in range(n):
        if valid[k]:
            last = (k, theta[k], omega[k])
        src = k + HORIZON
        if last is not None and src < n:
            j, th0, om0 = last
            tau = (src - j) * dt
            pred[src] = th0 * math.cos(w * tau) + om0 / w * math.sin(w * tau)
    pred[:HORIZON] = theta[:HORIZON]  # no forecast yet: show the AI on top of the sensor
    return pred


def referee(measured: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, int]:
    """Stand-in score in [0, 1] and the first flagged step (-1 if never)."""
    score, ema = np.zeros(len(pred)), 0.0
    above, first = 0, -1
    for k in range(len(pred)):
        if valid[k]:
            ema = 0.85 * ema + 0.15 * abs(measured[k] - pred[k])
        score[k] = 1.0 - math.exp(-ema / 0.06)
        above = above + 1 if score[k] > FLAG_AT else 0
        if above >= FLAG_HOLD and first < 0:
            first = k
    return score, first


def r4(a) -> list:
    return [None if (x is None or not np.isfinite(x)) else round(float(x), 4) for x in a]


def main() -> None:
    cfg = load_config(ROOT / "configs/pendulum.yaml")
    times = integrate.time_grid(cfg["dt"], cfg["n_steps"])
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for sid, title, regime, ptype, subtype, severity, blurb in SCENARIOS:
        onset = -1 if ptype == "none" else ONSET
        run = simulate_run(cfg, RunSpec(0, "demo", regime, ptype, subtype, severity, onset), np.random.default_rng(SEED))
        g, L = (float(v) for v in run["nominal_params"][:2])
        true, meas, mask = run["true_states"], run["measured_states"], run["measured_mask"]
        valid = mask[:, 0] == 1
        pred = ai_forecast(meas[:, 0], meas[:, 1], valid, cfg["dt"], g, L)
        score, flag = referee(meas[:, 0], pred, valid)
        answer = expected_verdict(regime, ptype)
        flagged_as = answer if answer != "trust" else "model_error"  # a stand-in false alarm would say this
        verdict = ["trust" if (flag < 0 or k < flag) else flagged_as for k in range(len(times))]
        conf = np.clip(0.95 - 0.04 * np.abs(meas[:, 0]), 0.6, 0.97)
        conf[~valid] = np.nan
        replay = {
            "id": sid, "title": title, "description": blurb,
            "is_demo_data": True,
            "system": "pendulum", "L": L, "dt": cfg["dt"],
            "onset_step": onset,
            "perturbation": {"type": ptype, "subtype": subtype, "severity": severity},
            "expected_verdict": answer,
            "frames": {
                "t": r4(times),
                "true_theta": r4(true[:, 0]),
                "measured_theta": r4(np.where(valid, meas[:, 0], np.nan)),
                "pred_theta": r4(pred),
                "ai_confidence": r4(conf),
                "referee_score": r4(score),
                "verdict": verdict,
                "evidence": [None if v == "trust" else EVIDENCE[v] for v in verdict],
            },
        }
        (OUT / f"{sid}.json").write_text(json.dumps(replay, separators=(",", ":")))
        missed = flag < 0 and answer != "trust"
        if missed:
            replay["title"] = title = f"{title} (missed)"
            (OUT / f"{sid}.json").write_text(json.dumps(replay, separators=(",", ":")))
        outcome = "no flag" if flag < 0 else f"flags {flagged_as} at step {flag}"
        print(f"{sid:22s} answer={answer:16s} {outcome}")
        index.append({"id": sid, "title": title, "file": f"{sid}.json", "expected_verdict": answer, "missed": missed,
                      "selection_rule": f"DEMO DATA: fixed seed {SEED}, onset step {ONSET}; not selected from results"})
    (OUT / "index.json").write_text(json.dumps({
        "generated_from": "demo/scripts/make_demo_replays.py (stand-in AI + referee, not real results)",
        "is_demo_data": True, "scenarios": index}, indent=2))
    print(f"wrote {len(index)} replays + index.json to {OUT}")


if __name__ == "__main__":
    main()
