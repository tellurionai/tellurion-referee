# Tellurion Referee: Hackathon Execution Plan

**Project:** Detect when a neural world model disagrees with reality, and diagnose *why*: model error, instrument fault, or discovery.
**Event:** AI for physical sciences / mathematics hackathon, Oct 21–22, 2026
**Submission deadline:** Thu Oct 22, 7:00 p.m. CT / 5:00 p.m. PT / 8:00 p.m. ET (confirmed in the organizers' announcement)
**Schedule:** kickoff **Wed Oct 21, 8:00 a.m. ET** (anywhere-on-Earth start). **Tuesday is prep only:** plans, formats, design, empty repo + CI. No project code before kickoff. Tickets on the GitHub board are the source of truth for dates.
**Status:** Implementation spec. Not a claim of experimental success.

> **North star:** *Detect when a model's predictions disagree with reality, and say why: model error, instrument fault, or something new.*
> This hackathon demonstrates that sentence on a pendulum, with numbers.

---

## 0. Rules for all agents

1. **Implement, measure, report. Never invent results.** Every plot and claim is generated from committed outputs.
2. **One learned world model family** (residual MLP, ensemble of 5 seeds). No LLMs in the pipeline. The reference solver and the diagnoser are not world models.
3. **Freeze world-model weights** after training. Record the SHA256 of each checkpoint; verify it is unchanged at final evaluation.
4. **No future ground truth online.** At decision time `t`, the referee sees only: nominal parameters, measured states up to `t`, the measurement mask, and the model's predictions. Never `true_states`, `actual_params`, `perturbation_*` or `expected_verdict`. Labels are for training the diagnoser (train/val/cal-derived only) and for evaluation.
5. **Scientific validity over polish.** Every method is compared with simple baselines (§3.4).
6. **No leakage.** Splits come from the reality engine dataset by `trajectory_id`. `test` is locked: evaluated once, at Gate 3.
7. **Coordinate through interfaces.** Own your files. The data contract is `plan-reality-engine.md` §5; don't fork it.
8. **Log** environment, seeds, hyperparameters, hardware and timings. A CPU smoke config runs end-to-end in minutes.
9. **Negative results are reported, not hidden.** We don't tune perturbation strengths after seeing results.

---

## 1. Research question and preregistered hypotheses

**Primary question:** Can a physics-grounded referee detect when a frozen neural dynamics model's forecasts diverge from reality *earlier and more reliably than model uncertainty alone*, and correctly attribute the cause?

- **H1 (detection, primary):** On the locked test set, the referee detects disagreement with higher AUROC and shorter delay after onset than an ensemble-uncertainty baseline at the same false-alarm rate on clean in-distribution runs.
- **H2 (confidently wrong):** Among steps where forecast error exceeds τ while ensemble confidence is high, the referee flags a majority.
- **H3 (diagnosis):** The referee's 4-way verdict (`trust / model_error / instrument_fault / discovery`) beats majority-class and uncertainty-only baselines on macro-F1, reported per severity.
- **H4 (stretch, P1):** A calibrated TRUST/DEFER policy achieves lower risk at equal coverage than thresholding ensemble variance.
- **Null outcomes** (no gain over uncertainty, verdicts confused at low severity, etc.) are reportable findings.

**Do not claim:** mechanistic understanding, general physics support, or transfer to video/foundation world models.

---

## 2. Scope and cut lines

### P0: must ship
- `tellurion_sim` pendulum dataset consumed through the data contract.
- World model: residual-MLP one-step model, **ensemble of 5 seeds**, trained on `train` (gentle, clean) only; frozen.
- Online forecaster: one-step and k-step predictions from *measured* states.
- **Referee v0 (rule-based, physics-grounded)**: physics checks → disagreement score + verdict (§3.3).
- Baselines (§3.4), locked-test evaluation with bootstrap CIs, one composite figure, README, 1-page summary, ≤4-min demo.

### P1: after P0 works end-to-end
- **Referee v1 (learned):** small classifier (logistic regression / gradient boosting) on referee features, trained on `cal`-derived examples with grouped CV, calibrated (isotonic or temperature) on a disjoint part of `cal`.
- **Conformal prediction** on top of the v1 classifier (split conformal, fit on a disjoint part of `cal`): output a *set* of verdicts with a coverage guarantee (e.g. "true cause ∈ {instrument_fault, model_error} with 90% coverage"). Report empirical coverage + average set size on test. An ambiguous set is a feature: it's the honest "can't tell which" answer.
- TRUST/DEFER policy + risk–coverage curve.
- **Not used:** third-party closed classifiers (e.g. Jev) for the verdict. The referee must be inspectable and self-calibrated; an optional side-by-side comparison is fine only if time is spare.
- Double pendulum results (if the reality engine ships it).

### P1 track: the inside view (mech interp, published methods only)
Goal: show the full Tellurion story, outside view **and** inside view, on the same frozen world model. Starts only after G2 (referee end-to-end on `cal`). Areas: physics fit + model. Files: `src/referee/inside.py`, `scripts/inside_view.py`, `tests/test_inside.py`.
1. **Record activations** of every hidden layer of the frozen ensemble on `cal` and `test` rollouts (forward hooks; weights untouched, hash re-checked).
2. **Linear probes** (ridge regression, fit on clean `train` activations only): does each layer encode angle, angular velocity and energy? Report held-out R² per layer. This answers "does the model represent the physics?"
3. **Activation-distance early warning:** Mahalanobis distance of activations from the clean `train` distribution (fit on `train` only; threshold set on `cal` at the same ≤5% false-alarm rate as the referee). Flag when distance crosses the threshold.
4. **The finding:** on `test`, compare detection delay (steps after onset) of (a) activation distance vs (b) the referee's outside-view checks vs (c) ensemble confidence. Report per perturbation type: the inside view should help most on "AI is wrong" cases (out-of-distribution states) and least on sensor faults the model never sees as unusual.
5. **Optional: physics-faithfulness score** with the published ARC ruler (floor = memorizer / persistence, ceiling = the reference simulator) on the hackathon model, if time allows.
6. **Output:** one panel for Figure 1 (detection delay by method and perturbation type) + one line in the one-pager. Same rules: bootstrap CIs by trajectory, locked test evaluated once, no tuning on test.

**Novelty this adds (a finding, not a new method):** "On a learned world model, simple internal checks flag trouble before its predictions fail, and help tell model errors apart from sensor faults."

**Out of scope:** anything beyond the published probing and activation-distance methods above.

### P2 / not in scope (public)
Video/foundation models, UI, RL controllers, LLM agents, and any method not listed in this plan.

**Hard rule:** if P0 locked-test results don't exist by **Thu Oct 22, 12:00 p.m. ET**, stop all features; ship evaluation, figure, docs.

---

## 3. System design

```text
 tellurion_sim ──► measured states + mask + nominal params
                                   │
                      Frozen world model (5-seed ensemble)
                                   │  one-step / k-step forecasts, ensemble spread
                                   ▼
        ┌──────────────────── REFEREE ────────────────────┐
        │  Physics checks (no learned components):         │
        │   1. forecast residual  (model vs measured)      │
        │   2. sensor self-consistency (θ vs ω kinematics, │
        │      jumps, stuck, dropout)                      │
        │   3. nominal-physics fit (reference solver)      │
        │   4. refit-physics fit (L, g, b re-estimated)    │
        │   5. energy residual (ideal-physics runs only)   │
        └─────────────► disagreement score + verdict ──────┘
                                   │
                    TRUST / DEFER  (+ verdict, + evidence)
                                   │
                    Evaluation vs. labels on locked test
```

### 3.1 Data
Use the reality engine's data contract exactly (`plan-reality-engine.md` §5): `true_states`, `measured_states`, `measured_mask`, `nominal_params`, `actual_params`, `split`, `regime`, `perturbation_*`, `onset_step`, `expected_verdict`.
- `train`/`val` = gentle + clean only. The world model trains on `train`, early-stops on `val`. **It trains on `measured_states`** (clean in these splits, so equal to truth).
- `cal` = labelled mix, used for referee thresholds, the v1 classifier and calibration.
- `test` = labelled mix, locked.

**Sensor assumption (confirm with the reality engine work):** both θ and ω are measured; sensor faults corrupt **θ only**. This is what makes the kinematic check (3.3.2) informative. If the reality engine changes it, update 3.3.2.

### 3.2 World model
- Residual MLP: input `[θ, ω]` (wrap-safe: use `[sin θ, cos θ, ω]`), output Δstate. 2–3 hidden layers × 64–128, fixed normalization from `train`.
- 5 seeds → ensemble mean = forecast; ensemble std = "uncertainty". **This is the standard uncertainty baseline.**
- Rollouts: autoregressive from the last measured state; never fed `true_states`.

### 3.3 Referee v0: verdict logic
Computed online over a sliding window W (e.g. 20 steps) ending at `t`:

1. **Forecast residual** `r_t = ‖measured_t − f(measured_{t−1})‖` (normalized). Large `r` → *something* disagrees. This drives detection.
2. **Sensor self-consistency.** Compare finite-difference `Δθ/dt` with measured ω (trapezoidal). Also: step jumps beyond physical max, zero variance (stuck), mask gaps. Inconsistent → **instrument_fault**.
3. **Nominal-physics fit.** Re-simulate the window from its first measured state with the reference ODE + nominal params. Fits well while the learned model doesn't → **model_error** (model out of its training range).
4. **Refit-physics fit.** Least-squares re-estimate `(L, g, b)` on the window. Nominal fails but a refit fits → **model_error** (subtype: parameter shift; report the estimated params, which is useful evidence).
5. **Unexplained structure.** Sensors consistent, but neither nominal nor refit physics explains the window → **discovery** (subtype hint from residual shape: constant / periodic / impulsive).
6. Nothing exceeds thresholds → **trust**.

Thresholds are set on `cal` only (target ≤5% false alarms on clean gentle `cal` runs). The referee outputs `{score, verdict, evidence}` per step, where evidence = which checks fired and their values. Evidence is part of the product: "why", not just "what".

### 3.4 Baselines (all use the same inputs and thresholds tuned on `cal`)
- **B1 Ensemble uncertainty:** flag when ensemble std > threshold. Detection only, can't diagnose (verdict = model_error when flagged).
- **B2 Residual threshold:** flag when `r_t` > threshold. Detection only.
- **B3 Majority / prior verdict:** for diagnosis.
- **Oracle reference (reported for context, not compared against):** detection using `true_states`, to show the ceiling.

### 3.5 Metrics (fixed before runs)
- **Detection:** AUROC (per step, aggregated by trajectory), **detection delay** = steps from `onset_step` to first sustained flag (3 consecutive), **false-alarm rate** on clean in-distribution runs, miss rate.
- **Confidently wrong:** fraction of steps with forecast error > τ (τ = 0.25 normalized) *and* ensemble confidence above its 80th percentile on clean data; the referee's recall on those steps vs B1.
- **Diagnosis:** 4×4 confusion matrix, macro-F1, per-class recall, broken down by severity and perturbation subtype.
- **Calibration (P1):** Brier, ECE + reliability diagram, risk–coverage for TRUST/DEFER.
- **Forecast quality context:** reliability horizon of the world model per regime (first sustained crossing of τ).
- **Cost:** referee wall time per step vs model inference.
- Report N, medians, and bootstrap 95% CIs **resampled by trajectory**.

---

## 4. Module interfaces

```python
# provided by the reality engine
from tellurion_sim.dataset import load_dataset          # -> dict of arrays (contract §5)
from tellurion_sim.systems.pendulum import Pendulum     # reference ODE, energy

# model.py
class DynamicsModel(torch.nn.Module): ...
class Ensemble: forecast(x, k) -> (mean [B,k,2], std [B,k,2])

# referee.py
class Referee:
    def fit_thresholds(self, cal_data) -> None
    def step(self, history: dict) -> dict   # {"score", "verdict", "evidence"}
# history contains ONLY: measured_states[:t+1], mask[:t+1], nominal_params, model forecasts

# features.py   (shared by referee v0 and v1)
def window_features(history, ensemble, system) -> dict

# diagnose_learned.py   (P1)
fit(features, labels, groups) / calibrate(model, cal_features, cal_labels) / predict_proba(...)

# evaluate.py
evaluate(method, split="test") -> dict   # writes results/*.json, results/*.csv
```

A unit test asserts `Referee.step` never receives a key from {`true_states`, `actual_params`, `perturbation_type`, `perturbation_subtype`, `severity`, `onset_step`, `expected_verdict`}.

---

## 5. Work areas and files

**Core work areas:** referee (the verdict logic), world model, evaluation + exporter, and figures + README + results fact sheet. Anyone on the team can pick up any area; the file list for each area keeps changes from colliding. The team writes the one-page summary from the results fact sheet.

### Reality engine
Covers `tellurion_sim/` per `plan-reality-engine.md`. Hands off: data contract **Tue evening** (spec only); clean gentle train/val **Wed 11:00 a.m. ET**; full labelled `pendulum_v1` **Wed 3:00 p.m. ET**.

### Agent A: World model
**Files:** `src/referee/model.py`, `scripts/train.py`, `tests/test_model.py`.
- Train a 5-seed ensemble on `train`; early stopping on `val`; save checkpoints, normalization, SHA256.
- Report one-step error and reliability horizon per regime (gentle / large / rotating).
- Until the reality engine data lands, train on a stub generator with the same schema; swap without code changes.
**Acceptance:** CPU-loadable; deterministic by seed; meaningful in-distribution accuracy *and* visible OOD failure (that's the point).

### Agent B: Referee (core)
**Files:** `src/referee/referee.py`, `features.py`, `diagnose_learned.py` (P1), `tests/test_referee.py`.
- Implement checks 3.3.1–3.3.5, threshold fitting on `cal`, `{score, verdict, evidence}` output.
- Unit tests on hand-built cases: a pure bias case → instrument_fault; a clean large swing → model_error; a magnet case → discovery; a clean gentle run → trust.
**Acceptance:** leakage test passes; deterministic; every verdict carries evidence.

### Agent C: Evaluation, baselines, calibration
**Files:** `src/referee/evaluate.py`, `baselines.py`, `scripts/evaluate.py`, `tests/test_evaluate.py`.
- Implement §3.4 baselines and §3.5 metrics exactly; trajectory-level bootstrap.
- (P1) Calibration + TRUST/DEFER risk–coverage.
**Acceptance:** machine-readable results; identical trajectories for all methods; test evaluated once (logged).

### Agent D: Figures, README, results fact sheet
**Files:** `scripts/make_figures.py`, `README.md`, `docs/results_fact_sheet.md`, `docs/references.md`, `figures/`.
**Not in this area:** `docs/one_page_summary.md` and `docs/demo_script.md` (the team writes these).
- **Figure 1:** (A) architecture; (B) one test trajectory (selection rule stated, e.g. "median-delay magnet case") showing reality vs model, ensemble confidence still high, referee flag + verdict + evidence; (C) detection delay / AUROC vs B1 with CIs, plus the 4×4 confusion matrix.
- Develop with `DEMO_ONLY` fake data; final figures only from `results/`.
- **Results fact sheet** (`docs/results_fact_sheet.md`, auto-generated from `results/` by a script): every headline number with its CI and N, the confusion matrix, per-severity table, selection rule for the example case, known failures. The one-pager is written **from this sheet**, so no number is typed by hand.
- **README:** install, exact commands, what each folder is, how to reproduce Figure 1 (Platform verifies it on a clean machine).
- The interactive web demo and the video are covered in `plan-demo.md`. Agent D supplies the figure and the fact sheet.

### Exporter for the demo (Agent C)
- `scripts/export_replays.py`: reads `results/` + the reality engine dataset and writes `demo/public/replays/` in the format of `demo/REPLAY_FORMAT.md` (per-frame true/measured/predicted θ, AI confidence, referee score, verdict, evidence). One command, no hand edits. Agree the format with the demo side by Tue evening (spec only); re-export from the locked test results right after G3.

### Physics-fit checks for the referee (Core)
Background: builds and fits digital cell models (differential-equation models fitted to data), which is the same skill as referee checks 3.3.3 and 3.3.4.
- **Files:** `src/referee/physics_fit.py` + `tests/test_physics_fit.py`:
  - **Nominal-physics fit (3.3.3):** re-simulate a window from its first measured state with the reality engine's reference ODE + nominal params; return fit error.
  - **Refit-physics fit (3.3.4):** least-squares re-estimate of `(L, g, b)` on the window (`scipy.optimize.least_squares`), with bounds; return the fitted params, their uncertainty, and fit error.
  - Must be fast enough to run per window during evaluation (cache, warm-start from the previous window).
- **Unit tests:** a clean window → nominal fit good; a longer-string window → refit recovers L within tolerance; a magnet window → neither fits.
- The referee area wires these outputs into the verdict logic; the physics-fit area doesn't change the verdict rules.
- **Thu 12–4 p.m.:** independent number check: every number in the one-pager, the figure, demo and video script matches `results/` exactly.
- **IP rule:** nothing from the professor's project (code, data) goes into Tellurion.

### Referee decisions
Scientific decisions, `configs/base.yaml`, scope cuts, merge approvals for `src/referee/`, final go/no-go at G3.

### Platform / integration
Repo skeleton, `pyproject.toml`, `Makefile`, CLAUDE.md/AGENTS.md, CI (tests + smoke + secret scan + leakage guard), branch protection, demo auto-deploy, test-commit tag, clean-machine reproducibility check, release and public-link check. See `plan-platform.md`.

### Demo
Web demo + video. See `plan-demo.md`.

---

## 6. Repo layout (public)

```
tellurion-referee/
├── README.md  LICENSE (Apache-2.0)  DATA_LICENSE (CC BY 4.0)
├── pyproject.toml  Makefile  configs/base.yaml
├── tellurion_sim/          # reality engine
├── demo/                   # web demo + video
├── src/referee/
│   ├── model.py  features.py  referee.py
│   ├── baselines.py  evaluate.py  diagnose_learned.py
├── scripts/ train.py  evaluate.py  make_figures.py
├── tests/
├── results/  figures/
└── docs/ one_page_summary.md  demo_script.md  references.md
```

---

## 7. Commands

```bash
pip install -e '.[dev]'
python -m tellurion_sim generate --system pendulum --config configs/pendulum.yaml --seed 0 --out data/pendulum_v1
python scripts/train.py --config configs/base.yaml
python scripts/evaluate.py --config configs/base.yaml --split cal     # thresholds + dev
python scripts/evaluate.py --config configs/base.yaml --split test    # ONCE, at Gate 3
python scripts/make_figures.py --input results/test_metrics.json
pytest -q
```

---

## 8. Gates and timeline (ET)

| When | Gate | Go / no-go |
|---|---|---|
| **Tue Oct 20, eve** | G0 (prep, no project code): repo skeleton + CI, data + replay formats agreed, physics-fit test spec, related-work search | Formats not agreed → resolve at the kickoff call |
| **Wed Oct 21, 8:00 a.m.** | Kickoff: code starts | |
| **Wed Oct 21, 3:00 p.m.** | G1: ensemble trained (stub data until 11 a.m., then the reality engine's); labelled `pendulum_v1` in; referee checks unit-tested | No data at 11 a.m. → keep training on the stub |
| **Wed Oct 21, 11:00 p.m.** | G2: referee v0 + baselines end-to-end on `cal` | If diagnosis is weak, keep detection as the headline. Late → cut all stretch |
| **Thu Oct 22, 12:00** | G3: **freeze**; run locked test once | Hard stop on features |
| **Thu Oct 22, 12–4 p.m.** | G4: figure, 1-pager, demo video, public repo check | Submission beats features |
| **Thu Oct 22, ≤6:00 p.m.** | Submit (2 h buffer) | |

Kickoff is Wed 8:00 a.m. ET; that leaves ~36 working hours. Cut lines matter more than usual. We're confirming with the organizers that an empty repo + agreed interfaces before kickoff is fine.

---

## 9. Risks and pivots

| Risk | Symptom | Pivot |
|---|---|---|
| Model too good OOD | Little failure in large/rotating | Increase horizon; add double pendulum; never damage the model deliberately |
| Model explodes immediately OOD | Trivial detection | Report it; the story moves to *diagnosis* and to the low-severity perturbations |
| Low-severity faults undetectable | Recall ≈ 0 at `low` | Report per-severity; that's a real detection limit |
| Verdict confusion (param shift vs discovery) | Off-diagonal mass | Show the confusion matrix honestly; refit-physics evidence explains which cases are ambiguous |
| Referee only beats B1 on easy cases | Gains vanish at matched false-alarm rate | Report at matched FAR; that's the fair comparison |
| Leakage | Test used for thresholds | Discard results, regenerate test with a new seed, disclose |
| Overbuilding | No end-to-end run by Wed 11 p.m. | Cut P1, double pendulum, learned referee; ship v0 + B1 |

---

## 10. Definition of done
- [ ] the reality engine dataset consumed via the contract; splits verified.
- [ ] Frozen 5-seed ensemble with hashes; OOD failure documented.
- [ ] Referee v0 with evidence; leakage test passes.
- [ ] B1/B2/B3 baselines on the identical locked test set.
- [ ] Detection, confidently-wrong, and diagnosis metrics with trajectory-level CIs, including failures.
- [ ] Calibration claims only if measured (else none).
- [ ] Figure 1 from real results; 1-page summary; ≤4-min demo.
- [ ] Public repo (Apache-2.0 code, CC BY data), links work without login.
- [ ] Submitted before the deadline.

---

## 11. Framing and related work (verify, cite precisely)
1. Lakshminarayanan et al. (2017), *Deep Ensembles*: The uncertainty baseline. https://arxiv.org/abs/1612.01474
2. Chua et al. (2018), *PETS*: probabilistic dynamics ensembles. https://arxiv.org/abs/1805.12114
3. Janner et al. (2019), *When to Trust Your Model (MBPO)*. https://arxiv.org/abs/1906.08253
4. Greydanus et al. (2019), *Hamiltonian Neural Networks*. https://arxiv.org/abs/1906.01563
5. Raissi et al. (2019), *Physics-informed neural networks*. https://doi.org/10.1016/j.jcp.2018.10.045
6. **Classical FDI (sensor vs system faults, hand-built models):** NASA, *Modeling, Detection, and Disambiguation of Sensor Faults for Aerospace Applications*: https://ntrs.nasa.gov/api/citations/20090033812/downloads/20090033812.pdf · sensor-fault isolation in nuclear plants: https://hal.archives-ouvertes.fr/hal-00201029. Our sensor-consistency check *is* classical FDI; say so.
7. **Fault vs model drift:** *Distinguishing Process Faults from Model Drift Through Variable Contribution Analysis*, Processes (2026): https://doi.org/10.3390/pr14050859. Closest to our model-vs-reality split; industrial, no discovery class. **Read in full before writing.**
8. **Sensor faults with learned dynamics:** de Silva et al., *Physics-informed ML for sensor fault detection with flight test data* (UW + Boeing). Real manoeuvres show up as false alarms: The confusion we resolve.
9. **World-model physics benchmarks (model assumed at fault, clean data):** World Models' Last Exam in Physics https://arxiv.org/abs/2610.08791 · WorldBench https://arxiv.org/abs/2601.21282 · MiraBench https://arxiv.org/abs/2605.29360 · WorldSimProbe https://arxiv.org/abs/2608.09298
10. **Self-driving-lab failures (no discovery vs fault split):** LabRobFail https://arxiv.org/abs/2607.23704
11. **Discovery vs detector artefacts (collider physics):** model-independent new-physics anomaly detection, e.g. https://arxiv.org/abs/1112.3329
12. **Calibration:** conformal prediction (Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction*, https://arxiv.org/abs/2107.07511).
13. **To check before writing (not yet reviewed):** Kennedy & O'Hagan (2001) Bayesian model discrepancy; "anomaly attribution" + digital twin. Still to check before any paper write-up.

**Careful novelty claim (use this wording):**
> "Fault diagnosis in engineering separates sensor faults from system faults, and recent world-model benchmarks measure *whether* models violate physics. To our knowledge, no prior work jointly attributes a learned world model's disagreement with reality to model error, instrument fault, or genuine novel physics, or provides a labelled benchmark for doing so."

Individual pieces (ensembles, residual FDI, parameter refitting, conformal prediction) are not new; don't claim they are. Framing: **we bring decades of fault-diagnosis discipline to learned world models, and add discovery as a first-class outcome.**

**If the inside-view track lands, add:** "...and the model's own internals raised the flag N steps before its predictions failed."

**Demo one-liner:** "The model is still 90% confident while it's already wrong. The referee caught it N steps after onset and says the sensor, not the physics, is at fault."
