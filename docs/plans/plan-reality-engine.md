# Reality Engine: Build Plan

**Area:** Reality engine (`tellurion_sim/`)
**Project:** Tellurion hackathon (AI for physical sciences, Oct 21–22, 2026)
**Submission deadline:** Thu Oct 22, 5:00 p.m. PT / 8:00 p.m. ET
**Schedule:** kickoff **Wed Oct 21, 8:00 a.m. ET** (anywhere-on-Earth start). **Tuesday is prep only:** plans, formats, design, empty repo + CI. No project code before kickoff. Tickets on the GitHub board are the source of truth for dates.
**Your piece in one line:** a physics engine that generates ground truth for a few mechanical systems, with switches that break reality in *labelled* ways, writing every result in one shared file format.

---

## 1. Why this exists

Tellurion builds AI that tells you **when a model's predictions disagree with reality, and why**. When they disagree, there are three possible causes:

| Verdict | Meaning | Example |
|---|---|---|
| **Model error** | The AI's physics is wrong for this situation | Trained on gentle swings, asked about a big swing |
| **Instrument fault** | The physics is fine, but the sensor is lying | Angle sensor reads +5° too high |
| **Discovery** | Something real is happening that nobody modelled | A magnet is pulling on the bob |

To prove our system names the right cause, we need test cases where **we already know the answer**. Your engine produces them. Every run is labelled with what we changed, so the rest of the team can check whether the diagnoser got it right.

You don't touch any ML. The Core team trains the AI model and the diagnoser on your data.

---

## 2. Scope

### P0: must ship by **Wed Oct 21, 3:00 p.m. ET**
- Single pendulum (with optional damping).
- All perturbation types in §4 for the pendulum.
- Dataset generator + data contract (§5) + tests (§7).
- CLI (§6).

### P1: after P0 is merged
- Double pendulum (chaotic; good for "model error" cases).
- Cart-pole (unactuated, or with a recorded random force input).

### Not yours: playback / demo
The web demo and video are built by the demo side (`plan-demo.md`). Your only demo duty is that `export.py` and `metadata.json` give them what they need (true and measured angles, onset, labels).

### Out of scope this week
ML models, Roblox, UI, rigid-body/contact engines, 3D. Keep it a clean Python library.

---

## 3. Physics core

**Stack:** Python 3.11, `numpy`, `scipy`. No torch dependency.

**Integrator:** `scipy.integrate.solve_ivp`, method `DOP853`, `rtol=1e-9`, `atol=1e-11`, with output sampled on a fixed grid (`dt = 0.05 s` default, configurable). Store solver settings in metadata. For discontinuous events (impulses, onsets), split the integration at the event time; don't integrate across a jump.

**System interface** (every system implements this):

```python
class System:
    name: str
    state_names: list[str]          # e.g. ["theta", "omega"]
    param_names: list[str]          # e.g. ["g", "L", "m", "b"]
    def rhs(self, t, x, p, force_fn=None) -> np.ndarray: ...   # dx/dt
    def energy(self, x, p) -> float: ...                       # total mechanical energy
    def positions(self, x, p) -> dict[str, np.ndarray]: ...    # x/y of each body for playback
    def sample_initial_state(self, rng, regime: str) -> np.ndarray: ...
```

**Pendulum:**
```
dtheta/dt = omega
domega/dt = -(g/L) sin(theta) - (b/(m L^2)) omega + tau_ext(t, theta, omega)/(m L^2)
E = 0.5 m L^2 omega^2 + m g L (1 - cos theta)
```
Angle convention: theta = 0 hanging straight down, positive counter-clockwise. Bob position: `x = L sin(theta)`, `y = -L cos(theta)` with the pivot at (0, 0).

**Regimes** (initial-condition ranges; the model trains on `gentle`, so the others test out-of-distribution):
- `gentle`: |theta0| ≤ 0.6 rad, |omega0| ≤ 0.8 rad/s
- `large`: |theta0| in [1.5, 2.8] rad
- `rotating`: enough energy to go over the top

Nominal parameters: `g = 9.81, L = 1.0, m = 1.0, b = 0.0`.

---

## 4. Perturbations (the "knobs")

Each perturbation has a **type**, a **severity** (`none | low | med | high`, mapped to numbers in config), and an **onset step**: The step where it switches on. Before onset, the run is identical to the clean run. Onset lets us measure *how fast* the diagnoser notices. Default: onset drawn uniformly from steps 40–120 of 200; `onset=0` means "from the start."

### 4.1 `param_shift` → expected verdict: **model error**
Reality's parameters differ from the nominal ones the model was trained on.
- `L` ×(1.05 / 1.15 / 1.30)
- `g` ×(0.95 / 0.85 / 0.70)
- `b` from 0 to (0.05 / 0.15 / 0.40)

Note: in the ideal pendulum `m` cancels out. Changing only `m` must not change the trajectory (that's a test, not a perturbation).

### 4.2 `sensor_fault` → expected verdict: **instrument fault**
**Only the measured signal changes; the true state is untouched.** This is the most important rule in the file.
- `bias`: constant offset on theta (0.03 / 0.08 / 0.20 rad)
- `drift`: offset grows linearly after onset
- `noise`: Gaussian, std (0.01 / 0.03 / 0.08 rad)
- `scale`: gain error on theta (×1.03 / 1.10 / 1.25)
- `stuck`: sensor freezes at its last value after onset
- `dropout`: fraction of steps missing (5% / 20% / 50%) → value NaN, mask = 0

### 4.3 `unseen_force` → expected verdict: **discovery**
A real physical effect not in the nominal equations, added through `tau_ext`.
- `constant_torque`: steady push (wind / tilted mount)
- `magnet`: attraction toward angle phi_m, torque = k·sin(phi_m − theta)
- `periodic_drive`: A·cos(w·t) (shaking pivot approximation)
- `impulse`: one-off kick at onset (split integration at the event)
- `wall` *(P1)*: hard stop at theta_w with coefficient of restitution e (event detection in `solve_ivp`)

### 4.4 `none` → expected verdict: **trust**
Clean runs, needed as negatives. A diagnoser that flags everything is useless.

Every run records `expected_verdict` ∈ {`trust`, `model_error`, `instrument_fault`, `discovery`}.
For `none` runs in the `large`/`rotating` regimes, `expected_verdict = model_error` (the model is out of its training range even though reality is clean). Encode this rule in one function so it's consistent.

---

## 5. Data contract (the shared format)

**Everyone else builds against this. Write it as `docs/DATA_CONTRACT.md` and share it with the team by **Tue Oct 20, evening** (writing the spec is prep, not code). Change it only by announcing it there.**

### 5.1 Dataset file: `data/<name>.npz` + `data/<name>.metadata.json`

| Array | Shape / dtype | Meaning |
|---|---|---|
| `times` | `[T+1]` float64 | seconds |
| `true_states` | `[N, T+1, S]` float32 | what really happened |
| `measured_states` | `[N, T+1, S]` float32 | what the sensor reported (= true unless `sensor_fault`) |
| `measured_mask` | `[N, T+1, S]` uint8 | 1 = valid reading, 0 = dropped |
| `true_energy` | `[N, T+1]` float32 | from `system.energy` |
| `nominal_params` | `[N, P]` float32 | what the model is told / trained with |
| `actual_params` | `[N, P]` float32 | what reality used |
| `trajectory_id` | `[N]` int64 | unique, stable across regenerations with the same seed |
| `split` | `[N]` str | `train`, `val`, `cal`, `test` |
| `regime` | `[N]` str | `gentle`, `large`, `rotating` |
| `perturbation_type` | `[N]` str | `none`, `param_shift`, `sensor_fault`, `unseen_force` |
| `perturbation_subtype` | `[N]` str | e.g. `bias`, `magnet` |
| `severity` | `[N]` str | `none`, `low`, `med`, `high` |
| `onset_step` | `[N]` int32 | -1 if none |
| `expected_verdict` | `[N]` str | `trust`, `model_error`, `instrument_fault`, `discovery` |

`metadata.json`: system name, state/param names and units, dt, horizon, solver + tolerances, seed, git commit, severity→number table, split rules, generation time.

### 5.2 Split rules (no leakage)
- Split by `trajectory_id`, seeded, never by time step.
- `train` and `val`: **only** `gentle` regime, `perturbation_type = none`. This is what the AI model learns from.
- `cal` and `test`: a balanced mix of all regimes × perturbation types × severities, plus clean `gentle` runs.
- `test` is locked: nobody tunes on it.

Default sizes for the pendulum: 2,000 train / 400 val / 600 cal / 1,000 test, 200 steps each. Generation must take minutes on a laptop CPU.

### 5.3 Single-run CSV export (for playback and the demo)
Same columns as the earlier demo CSV, plus measurement and labels:
```
time_s, true_angle_rad, measured_angle_rad, measured_valid, predicted_angle_rad,
confidence_0to1, true_x, true_y, pred_x, pred_y,
perturbation_type, perturbation_subtype, severity, onset_step, expected_verdict
```
`predicted_angle_rad` and `confidence_0to1` stay empty in your export. The model team fills them in.

---

## 6. CLI

```bash
pip install -e '.[dev]'

# full dataset
python -m tellurion_sim generate --system pendulum --config configs/pendulum.yaml --seed 0 --out data/pendulum_v1

# one labelled scenario (for debugging and demos)
python -m tellurion_sim scenario --system pendulum --regime gentle \
    --perturb sensor_fault --subtype bias --severity med --onset 80 --seed 3 \
    --csv out/pendulum_bias_med.csv

# sanity report: energy drift, period check, label counts per split
python -m tellurion_sim check data/pendulum_v1
```
Every command takes `--seed`, writes nothing outside `--out`/`--csv`, and exits non-zero on failure.

---

## 7. Tests (`pytest -q`; must pass before handing off)

1. **Determinism:** same seed → byte-identical arrays.
2. **Energy conservation:** ideal pendulum (b = 0, no force), relative energy drift < 1e-6 over the horizon.
3. **Small-angle period:** theta0 = 0.05 rad → period within 0.1% of 2π√(L/g).
4. **Mass invariance:** ideal pendulum trajectory unchanged when only `m` changes.
5. **Zero severity = clean:** every perturbation at strength 0 reproduces the clean run exactly.
6. **Sensor faults never touch truth:** `true_states` identical with and without any `sensor_fault`.
7. **Onset:** before `onset_step`, the perturbed run matches the clean run exactly.
8. **Damping:** with b > 0, energy is non-increasing (within tolerance).
9. **Split hygiene:** train/val contain only `gentle` + `none`; no `trajectory_id` appears in two splits.
10. **Schema:** every array has the shape and dtype in §5.1; `expected_verdict` follows the §4.4 rule.

---

## 8. Repo layout (your files)

```
tellurion_sim/
├── __init__.py
├── __main__.py          # CLI
├── systems/
│   ├── base.py          # System interface
│   ├── pendulum.py      # P0
│   ├── double_pendulum.py  # P1
│   └── cartpole.py      # P1
├── perturbations.py     # param_shift, sensor_fault, unseen_force, labels
├── integrate.py         # solve_ivp wrapper, event/onset splitting
├── dataset.py           # generation, splits, npz + metadata
└── export.py            # single-run CSV
configs/pendulum.yaml
tests/test_*.py
```
You own everything in `tellurion_sim/`, its configs and its tests. The model team imports your package; nobody else edits it without asking.

---

## 9. Timeline

| When | Deliverable |
|---|---|
| **Tue Oct 20, evening** | Prep only: `DATA_CONTRACT.md` posted and acknowledged; confirm θ-only sensor faults with the Core team. No code |
| **Wed Oct 21, 8:00 a.m. ET** | Kickoff: start pendulum + integrator + tests |
| **Wed Oct 21, 11:00 a.m. ET** | Clean `gentle` train/val handed to the model work (so the model can train) |
| **Wed Oct 21, 3:00 p.m. ET** | P0 done: all perturbations, tests pass, full labelled `pendulum_v1` handed off |
| **Wed Oct 21, evening** | P1 (stretch): double pendulum, only if the team's end-to-end run is on track |
| **Thu Oct 22, morning** | Bug fixes; help the demo side with any scenario they need regenerated |
| **Thu Oct 22, 2:00 p.m. ET** | Code freeze for your package |

If you're behind on the Wednesday 3 p.m. target, cut P1. A correct pendulum with all perturbations beats three half-working systems.

---

## 10. Rules

- **Correctness over features.** If a test fails, fix it before adding anything.
- **This package will be public** (Apache 2.0) as part of our benchmark. Write it like other people will read it: docstrings, units in names or comments, no secrets or keys in code.
- **Nothing hand-tuned to make results look good.** Perturbation strengths come from the config table, chosen before anyone sees model results.
- Questions about physics, format or priorities → ask the team.
