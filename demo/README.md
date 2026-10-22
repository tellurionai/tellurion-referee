# Demo

An interactive web demo of the referee: pick a scenario, watch reality, the AI's
prediction and the sensor reading, and see the referee flag and name the cause.
It is a static site built with Vite + TypeScript + Canvas 2D, with no runtime dependencies.

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

## Replay data

The format is in [REPLAY_FORMAT.md](REPLAY_FORMAT.md).

The files in `public/replays/` are currently **DEMO DATA**. They come from
`scripts/make_demo_replays.py`, using real `tellurion_sim` physics but a stand-in AI
(a linearised pendulum) and a stand-in referee. They exist only to build the interface:

```bash
uv run --with-requirements requirements.txt python demo/scripts/make_demo_replays.py   # from the repo root
```

The real replays come from `scripts/export_replays.py` (#39, T41), which overwrites
this folder with `is_demo_data: false`. No code changes are needed; the validator
rejects any file that doesn't match the format.

## Not done yet

- Results panel (P1): needs `results/test_metrics.json` from evaluation.
- "Break it yourself" mode (P1): needs a library of precomputed real scenarios.
- Auto-deploy (#35) belongs to the platform team.
