# Referee Demo: Build Plan (Job 3)

**Area:** Demo (`demo/`)
**Suggested split:** (1) design, stage and verdict card; (2) replay loading, timeline/scrubber, results panel and the video edit. Anyone on the team can pick up either part.
**Deploys:** automatic on every merge to `main` (see `plan-platform.md`)
**Project:** Tellurion hackathon (AI for physical sciences, Oct 21–22, 2026)
**Submission deadline:** Thu Oct 22, 8:00 p.m. ET (5:00 p.m. PT). Target submit by 6:00 p.m. ET.
**Schedule:** kickoff **Wed Oct 21, 8:00 a.m. ET** (anywhere-on-Earth start). **Tuesday is prep only:** plans, formats, design, empty repo + CI. No project code before kickoff. Tickets on the GitHub board are the source of truth for dates.
**Your piece in one line:** turn the team's real results into a polished, interactive web demo and a ≤4-minute video that anyone can understand in 30 seconds and share.

---

## 1. The story you are telling

> *AI that predicts the physical world is sometimes confidently wrong, and nobody notices. Tellurion's referee catches it and says why: The AI is wrong, the sensor is wrong, or reality is doing something new.*

Everything on screen serves one moment: **The AI's confidence meter is still high while its prediction is already wrong, and the referee blows the whistle and names the cause.** If a viewer remembers one thing, it's that.

Who builds what:
- **Reality engine:** the physics simulator ("reality") with switches that break it on purpose (`tellurion_sim`).
- **Core:** the AI world model and the referee, graded against the reality engine's answer key.
- **You:** make it visible. You don't compute physics or run AI. You **replay** files the others produce.

---

## 2. Deliverables

### P0: must ship
1. **Interactive web demo** (static site, public URL, works on phone and laptop):
   - A scenario picker: e.g. "Normal swing", "Big swing", "Broken sensor", "Hidden magnet", "Longer string".
   - The pendulum animation: **reality** (solid navy), **AI prediction** (brass ghost), **sensor reading** (small marker; only differs when the sensor is broken).
   - Two meters side by side: **"AI confidence"** (what others give you) vs **"Referee"** (disagreement score).
   - A **verdict card** that appears when the referee flags: `AI is wrong` / `Sensor is wrong` / `Something new` + one line of evidence (e.g. "angle and speed readings contradict each other").
   - Play / pause / scrub timeline, with a marker where the hidden switch flipped ("onset").
2. **Demo video, ≤4 minutes, normal speed** (hackathon requirement): screen recording of the web demo + voiceover following §6.
3. **30–45 s social cut** (vertical 9:16 and 16:9) for social media.

### P1: after P0 works
- **"Break it yourself"** mode: user picks system + switch + severity → loads the nearest precomputed scenario (still replay, not live AI).
- Shareable links per scenario (`/#scenario=magnet-med`).
- Results panel: The headline numbers from `results/test_metrics.json` (detection delay vs "AI confidence" baseline, confusion matrix) with a "how we measured this" link to the repo.

### P2: only if everything else is done
- Double pendulum scenarios (if the reality engine ships them).
- Running the world model in the browser (ONNX). Not needed; replay is fine.

### Out of scope
Roblox, backend servers, user accounts, analytics that collect personal data, 3D.

---

## 3. Data you consume (the "replay" format)

You never generate numbers. You read **replay files** exported from the team's pipeline. Until real ones exist, build against `DEMO_ONLY` files (the demo CSV `pendulum_model_vs_reality.csv`, converted), visibly watermarked "DEMO DATA".

### 3.1 `replays/index.json`
```json
{
  "generated_from": "results commit <sha>",
  "is_demo_data": false,
  "scenarios": [
    {"id": "magnet-med", "title": "Hidden magnet", "file": "magnet-med.json",
     "expected_verdict": "discovery", "selection_rule": "median detection delay among magnet/med test cases"}
  ]
}
```

### 3.2 `replays/<id>.json`
```json
{
  "id": "magnet-med",
  "system": "pendulum", "L": 1.0, "dt": 0.05,
  "onset_step": 80,
  "perturbation": {"type": "unseen_force", "subtype": "magnet", "severity": "med"},
  "expected_verdict": "discovery",
  "frames": {
    "t":              [0.0, 0.05, ...],
    "true_theta":     [...],
    "measured_theta": [...],          // null where the sensor dropped a reading
    "pred_theta":     [...],          // AI ensemble mean
    "ai_confidence":  [...],          // 0–1, from ensemble agreement
    "referee_score":  [...],          // 0–1
    "verdict":        ["trust", ..., "discovery", ...],
    "evidence":       [null, ..., "sensors agree; no physics fit explains motion", ...]
  }
}
```
Positions are computed client-side: `x = L sin θ`, `y = −L cos θ`, pivot at the top.

**Who produces these:** the Core team adds `scripts/export_replays.py` (one command, reads `results/` + the reality engine dataset, writes `replays/`). **Agree the format with the Core team by Tue evening** and share it with the team. The demo side writes the format doc; Core writes the exporter.

---

## 4. Design

**Brand:** navy `#14304f` (reality), brass `#c9a46a` (AI), warning red `#c0392b` (referee flag only), off-white `#f6f3ec` background, dark mode supported. One serif for headings (e.g. *Fraunces* / *Source Serif*), one clean sans for UI (e.g. *Inter*).

**Layout (desktop):** pendulum stage left (60%), meters + verdict card right. **Mobile:** stage on top, meters below, picker as horizontal chips.

**Motion rules:**
- Default playback: real time; a "slow-mo around the flag" toggle.
- The referee flag gets the one big animation in the demo (a pulse + verdict card slide-in). Nothing else competes with it.
- The AI ghost leaves a faint trail so divergence is visible even in a still screenshot.

**Plain-language labels only:** "AI prediction", "Reality", "Sensor reading", "AI confidence", "Referee". No "ensemble std", "residual", "OOD" on the main screen (put them in a "technical details" drawer).

**Stack (suggested):** Vite + TypeScript + Canvas 2D, no heavy frameworks. Host on GitHub Pages or Vercel from the public repo (`/demo`). First load < 1 MB, 60 fps on a mid-range phone.

---

## 5. Honesty rules (non-negotiable; this is our brand)

1. **Only real outputs in the final demo.** `DEMO_ONLY` data is watermarked and never appears in the video or public site.
2. **No hand-picking without saying so.** Each showcased scenario states its selection rule (e.g. "median case", not "best case"). The index lists the rule.
3. **Show at least one honest miss** in the scenario library (e.g. a low-severity fault the referee didn't catch), labelled as such. It makes the hits believable.
4. **Numbers on screen come straight from `results/`**, never typed by hand.
5. **Describe the model accurately:** "a small neural world model trained on gentle swings". Not a foundation model.
6. **Describe only what it does.** The demo detects and diagnoses disagreements; make no claims beyond that.

---

## 6. Video script skeleton (≤4:00)

| Time | Beat | On screen |
|---|---|---|
| 0:00–0:20 | Hook: "This AI is 90% confident. It's also wrong." | Big swing scenario at the flag moment, frozen |
| 0:20–0:50 | The problem: AI that predicts the physical world fails silently | Normal swing (AI and reality overlap) → big swing (ghost peels away, confidence stays high) |
| 0:50–1:40 | The referee + three causes | Broken sensor → "Sensor is wrong"; hidden magnet → "Something new"; longer string → "AI is wrong" |
| 1:40–2:40 | How we know it works: The answer key, results vs the confidence-only baseline, one honest miss | Results panel + confusion matrix |
| 2:40–3:20 | Why it matters: labs, robots, digital twins: different cause → different fix. One honest line on what's new: "Engineers have diagnosed faults in hand-built models for decades. We bring that to learned world models, and treat discovery as a result, not an error." | Simple graphic |
| 3:20–3:50 | Open benchmark + repo link + team | Repo page, team names |
| 3:50–4:00 | Mission line + Tellurion wordmark | "Make AI that acts in the physical world honest about when it's wrong." |

the team picks who records the voiceover. Script final by Thu 12:00 p.m. ET.

---

## 7. Timeline (ET)

| When | Deliverable |
|---|---|
| **Tue Oct 20, evening** | Prep only: replay format agreed + posted; design mockups (desktop + mobile) agreed by the team; brand tokens chosen. No app code |
| **Wed Oct 21, 8:00 a.m.** | Kickoff: app skeleton + `DEMO_ONLY` replay and the stage start in parallel |
| **Wed Oct 21, 3:00 p.m.** | Full P0 UI on demo data: picker, stage, meters, verdict card, scrubber, mobile layout |
| **Wed Oct 21, 11:00 p.m.** | First **real** replays from the `cal` split; UI fixed against real edge cases (dropped readings, early flags, misses) |
| **Thu Oct 22, 12:00 p.m.** | Team freezes + runs the locked test; exporter regenerates replays from test results. **No tuning; just re-export.** |
| **Thu Oct 22, 12–3 p.m.** | Record + edit video; results panel live; social cut |
| **Thu Oct 22, 4:00 p.m.** | Site deployed, public URL checked in a logged-out browser + on a phone |
| **Thu Oct 22, ≤6:00 p.m.** | Submission sent |

If behind at Wednesday 3 p.m.: cut P1 entirely, keep 4 scenarios, simplify meters to bars.

Tuesday is design, mockups and the format doc only; app code starts at kickoff.

---

## 8. Files you own

```
demo/
├── index.html
├── src/
│   ├── main.ts          # app shell, routing (#scenario=...)
│   ├── replay.ts        # loads + validates index.json / <id>.json
│   ├── stage.ts         # pendulum canvas: reality, AI ghost + trail, sensor marker
│   ├── meters.ts        # AI confidence vs referee
│   ├── verdict.ts       # verdict card + evidence
│   └── timeline.ts      # play/pause/scrub, onset marker
├── styles.css           # tokens: colours, type, dark mode
├── public/replays/      # exported by the pipeline (not hand-edited)
├── REPLAY_FORMAT.md     # The contract in §3
└── video/
    ├── script.md
    └── shotlist.md
```
Demo work doesn't edit `tellurion_sim/` (reality engine) or `src/referee/` (Core). Ask the team instead.

---

## 9. Definition of done
- [ ] Public URL works logged-out, on phone and laptop, 60 fps.
- [ ] ≥5 scenarios covering all three causes + a clean run + one honest miss, each with its selection rule.
- [ ] Verdict card shows cause + evidence; onset marker visible.
- [ ] All numbers and replays come from the pipeline; no `DEMO_ONLY` data in public.
- [ ] ≤4:00 video at normal speed + 30–45 s social cut.
- [ ] Repo `demo/README.md`: how to run locally, how to regenerate replays.
