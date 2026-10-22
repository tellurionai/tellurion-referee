# tellurion-referee: rules for people and AI coding agents

A referee that detects when a learned world model disagrees with reality and says why:
**model error**, **instrument (sensor) fault**, or **something new**. Single pendulum for the hackathon.

## Plans
- `docs/plans/plan-reality-engine.md`: simulator, perturbations, dataset, data contract
- `docs/plans/plan-referee.md`: world model, referee, baselines, evaluation
- `docs/plans/plan-demo.md`: web demo + video
- `docs/plans/plan-platform.md`: repo, CI, deploys, release

## Areas and folders
Work only inside the folders for your ticket's area. Need a change elsewhere? Open an issue or ask the team.

| Area | Folders / files |
|---|---|
| Reality engine | `tellurion_sim/`, `configs/pendulum.yaml`, `tests/test_sim*.py`, `docs/DATA_CONTRACT.md` |
| Model | `src/referee/model.py`, `scripts/train.py`, `tests/test_model.py` |
| Referee | `src/referee/referee.py`, `src/referee/features.py`, `configs/base.yaml`, `tests/test_referee.py` |
| Physics fit | `src/referee/physics_fit.py`, `tests/test_physics_fit.py` |
| Evaluation | `src/referee/evaluate.py`, `src/referee/baselines.py`, `scripts/evaluate.py`, `scripts/export_replays.py`, `tests/test_evaluate.py` |
| Docs & figures | `scripts/make_figures.py`, `README.md`, `docs/`, `figures/` |
| Demo | `demo/` |
| Platform | repo root files, `.github/`, `results/RERUNS.md` |

## Rules
1. **Never load the locked `test` split** outside `scripts/evaluate.py --split test`. It runs once, at Gate 3 (Thu 12:00 ET). `tests/test_no_test_leakage.py` enforces this.
2. **No secrets** (API keys, tokens, webhook URLs) in code, issues, commits or screenshots. Use environment variables; `.env` is git-ignored.
3. **Every change goes through a pull request** that says `Closes #<issue>`. `make test` must pass. A teammate reviews before merge.
4. **Results come from scripts only.** No hand-typed numbers, no hand-picked best cases. Record commit SHA, seed and config in every results file.
5. **Build only what's in the plans.** New method or feature? Ask the team first.

## For AI coding agents
- Read the issue you are working on: its plan excerpt and "Done when" list are the spec.
- Write tests for each "Done when" item first, then the code. Stop when `pytest -q` passes.
- Edit only the files listed for the ticket's area. Do not touch the test split, other areas, CI or secrets.
- Finish with a short summary of what you changed and anything you were unsure about.

## Commands
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: smaller CPU-only torch
pip install -e '.[dev]'              # core + tests (Python 3.11+)
# pip install -e '.[dev,eval]'      # also scikit-learn, matplotlib for metrics and figures
make test          # all tests, including the leakage guard
make smoke         # tiny end-to-end run (< 5 min)
make data          # generate data/pendulum_v1
make train         # train the world model ensemble
make eval-cal      # referee + baselines on the cal split
make figures       # figures from results/
# make eval-test   # ONCE, at Gate 3 only
```
