# Replay format

The contract between the pipeline (`scripts/export_replays.py`, T41) and the demo.
The demo never computes physics or runs the AI; it only replays these files.
`src/replay.ts` validates every file before it is drawn, and refuses to draw a file
that breaks a rule below.

Files live in `demo/public/replays/`. The exporter writes them; nobody hand-edits them.

## `index.json`

```json
{
  "generated_from": "results commit <sha>",
  "is_demo_data": false,
  "scenarios": [
    {
      "id": "magnet-med",
      "title": "Hidden magnet",
      "file": "magnet-med.json",
      "expected_verdict": "discovery",
      "selection_rule": "median detection delay among magnet/med test cases",
      "missed": false
    }
  ]
}
```

| Field | Rule |
|---|---|
| `is_demo_data` | `true` shows the DEMO DATA banner and watermark. Must be `false` for anything public. |
| `scenarios[].id` | Unique. Also the shareable link: `#scenario=<id>`. |
| `scenarios[].selection_rule` | **Required** (honesty rule). Say how the case was picked, e.g. "median case", never just "best case". |
| `scenarios[].missed` | Optional. `true` for an honest miss, i.e. the referee never flagged a faulty run. Include at least one. |

Scenarios appear in the picker in this order.

## `<id>.json`

```json
{
  "id": "magnet-med",
  "title": "Hidden magnet",
  "description": "At the switch, a hidden magnet starts pulling the bob sideways.",
  "is_demo_data": false,
  "system": "pendulum", "L": 1.0, "dt": 0.05,
  "onset_step": 80,
  "perturbation": {"type": "unseen_force", "subtype": "magnet", "severity": "med"},
  "expected_verdict": "discovery",
  "frames": {
    "t":              [0.0, 0.05, "..."],
    "true_theta":     ["..."],
    "measured_theta": ["...", null, "..."],
    "pred_theta":     ["..."],
    "ai_confidence":  ["..."],
    "referee_score":  ["..."],
    "verdict":        ["trust", "...", "discovery"],
    "evidence":       [null, "...", "sensors agree; no physics fit explains the motion"]
  }
}
```

| Field | Rule |
|---|---|
| `L`, `dt` | Positive numbers. The demo draws x = L sin θ, y = −L cos θ, pivot at the top. |
| `onset_step` | Frame where the hidden switch flips, or `-1` for none. |
| `expected_verdict`, `frames.verdict[]` | One of `trust`, `model_error`, `instrument_fault`, `discovery`. |
| `frames.*` | All arrays have the same length (≥ 2). `t` is strictly increasing. |
| `true_theta` | Numbers only (rad). |
| `measured_theta` | rad, or `null` where the sensor dropped a reading. |
| `pred_theta` | AI ensemble mean (rad), or `null` before the AI has a forecast. |
| `ai_confidence`, `referee_score` | 0–1, or `null`. |
| `evidence[]` | One plain-language line when the verdict is not `trust`, else `null`. |
| `title`, `description` | Optional; override the index title and add a one-line explanation. |

The referee "flags" at the first frame whose `verdict` is not `trust`. The verdict card
shows that verdict with its evidence, and checks it against `expected_verdict`.
