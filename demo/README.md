# Demo

An interactive web demo of the referee: pick a scenario, watch reality, the AI's
prediction and the sensor reading, and see the referee flag and name the cause.
It is a static site built with Vite + TypeScript + Canvas 2D, with no runtime dependencies.
The plan is in [`docs/plans/plan-demo.md`](../docs/plans/plan-demo.md).

## Run locally

```bash
cd demo
pnpm install
pnpm dev        # http://localhost:5173
pnpm test       # replay validator, timeline, verdict card, shipped replays
pnpm build      # static site in demo/dist (relative paths, works from any subpath)
```

## Features

- **Scenario picker** (chips; scrolls sideways on phones), with a shareable link per scenario: `#scenario=hidden-magnet`.
- **Stage**: reality (navy), the AI prediction as a brass ghost with a 1.5 s fading trail, and the sensor reading as a ring. When a reading is dropped, the stage shows "Sensor: no reading".
- **Timeline**: play/pause, scrubbing, a marker for the hidden switch (onset) and one for the referee flag. Plays in real time, with optional slow-mo around the flag.
  - Keys: Space to play or pause, ←/→ to step one frame, Shift+←/→ to step ten.
- **Meters**: "AI confidence" next to "Referee". Only the referee turns red, and only when it flags.
- **Verdict card**: the one big animation. It shows the cause (*AI is wrong / Sensor is wrong / Something new*) with one line of evidence, then the answer key. An unflagged run reveals its miss only at the end.
- **Technical details** drawer: raw θ values, residual, scores, the hidden switch and the selection rule.
- Dark mode, reduced-motion support, and a layout that works on phones.
- A **DEMO DATA** banner and watermark whenever `is_demo_data` is true.

## Live site

Every merge to `main` that touches `demo/` is tested and published to GitHub Pages by
`.github/workflows/demo.yml` (https://tellurionai.github.io/tellurion-referee/).

## Replay data

The format is in [REPLAY_FORMAT.md](REPLAY_FORMAT.md).

The files in `public/replays/` are **real results**: the shared world model and the referee
on the evaluation split, exported by `scripts/export_replays.py` (`make replays`, or
`make replays SPLIT=test` after the locked test). Every scenario is picked by a stated rule
(the median case of its kind), never by whether the referee got it right, so some verdicts
are wrong and one run is an honest miss. The answer key is shown beside each verdict.

`scripts/make_demo_replays.py` still generates the old stand-in **DEMO DATA** (a stand-in AI
and a referee that copies the answer key), for UI work only. Never publish it.

## Not done yet

- Results panel (P1): needs `results/test_metrics.json` from evaluation.
- "Break it yourself" mode (P1): needs a library of precomputed real scenarios.
