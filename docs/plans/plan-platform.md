# Platform & Infra: Build Plan

**Area:** Platform (repo, CI, deploys, release)
**Works with:** Core, reality engine, demo
**Your piece in one line:** make sure everything the team builds **runs, is reproducible, and is secure**, so everyone else can focus on the science.

---

## Part 1: Hackathon (Oct 21–22)

**Deadline:** Thu Oct 22, 8:00 p.m. ET (5:00 p.m. PT). Team target: submit by 6:00 p.m. ET.

You take over the **integrator** duties from the referee plan. You don't write referee, model or physics code. You make sure everyone else's code fits together and ships.

> **Schedule:** kickoff **Wed Oct 21, 8:00 a.m. ET**. Tuesday is prep only. An **empty** repo skeleton, CI on a hello-world test, licences, CLAUDE.md and the ticket board are setup, not project code (being confirmed with the organizers). No referee/model/physics/demo code before kickoff.

### 1.1 Repo skeleton (first thing, before anyone writes code)
- One public repo `tellurion-referee` under the Tellurion GitHub org, with the folder layout from `plan-referee.md` §6:
  `tellurion_sim/` (reality engine), `src/referee/` (Core), `demo/` (demo), `scripts/`, `tests/`, `configs/`, `results/`, `figures/`, `docs/`.
- `pyproject.toml` with pinned versions (Python 3.11; numpy, scipy, torch CPU, scikit-learn, pytest). `pip install -e '.[dev]'` must work on a clean machine.
- `Makefile` targets: `data`, `train`, `eval-cal`, `eval-test`, `figures`, `test`, `smoke`.
- `configs/smoke.yaml`: a tiny dataset + tiny model that runs end-to-end on CPU in **< 5 minutes**.
- `LICENSE` (Apache-2.0 for code), `DATA_LICENSE` (CC BY 4.0), `.gitignore` (data/, artifacts/, .venv, secrets).
- **`CLAUDE.md` + `AGENTS.md`** at the root (AGENTS.md can just say "see CLAUDE.md"). Contents:
  - which folders belong to which area; stay inside your ticket's area unless you ask first
  - links to the four plans
  - rules: never read the locked test split, never commit secrets, every change passes `make test`
  - the commands above

### 1.2 Automatic checks (CI, GitHub Actions)
On every pull request:
1. `pytest -q` (all teams' tests)
2. `make smoke` (end-to-end on the tiny config)
3. Lint/format (ruff), kept light; don't block on style
4. **Secret scanning** (GitHub secret scanning + push protection on)
5. **Leakage guard:** a test that fails if anything outside `scripts/evaluate.py --split test` loads the `test` split

Keep CI under ~5 minutes or people will route around it.

### 1.3 Branch rules
- `main` is protected: no direct pushes; PR + 1 review + green CI to merge.
- Small PRs, one folder at a time where possible.
- Each person works on their own branch. Running parallel AI coding sessions? Use separate **git worktrees**, one per folder.

### 1.4 Demo deployment
- Auto-deploy `demo/` to GitHub Pages or Vercel on every merge to `main`.
- A preview deploy per PR is a bonus.
- Check: loads logged-out, on a phone, first load < 1 MB.

### 1.5 Results hygiene
- `results/` and `figures/` are produced only by scripts; record the commit SHA, seed and config in every results file.
- After the locked test run (Thu noon), **tag** the commit: `v0.1-hackathon-test`. Any later rerun is logged in `results/RERUNS.md`.

### 1.6 Reproducibility check (Thu ~2 p.m. ET)
On a **clean machine or fresh container**: clone → install → run every README command exactly as written → confirm the figures match. File anything that breaks as a blocker. Judges will do this.

### 1.7 Submission release (Thu 4–6 p.m. ET)
- Final tag `v0.1`, GitHub Release with a short changelog.
- Repo public; README links work logged-out; demo URL and video link work.
- Put a checklist of all links for the submission form in the release notes.

### Hackathon timeline (ET)
| When | Deliverable |
|---|---|
| **Tue Oct 20, evening** | Empty repo skeleton, CLAUDE.md, CI on a hello-world test, branch protection, Pages/Vercel connected, tickets imported (`create_tickets.py`) |
| **Wed Oct 21, 8:00 a.m.** | Kickoff: team starts coding against your skeleton |
| **Wed Oct 21, 3:00 p.m.** | `make smoke` green with the simulator package + the trained model; leakage guard in CI; demo auto-deploys |
| **Wed Oct 21, 11:00 p.m.** | Full pipeline runs in CI on the smoke config |
| **Thu Oct 22, 12:00** | Tag the locked test commit |
| **Thu Oct 22, 2:00** | Clean-machine reproducibility check |
| **Thu Oct 22, 4–6 p.m.** | Release, link checks, submission links ready |

---

## Rules
- **Boring and reliable beats clever.** No Kubernetes, server swarms or custom platforms until there's a real need.
- Never put secrets in code, issues, screenshots or chat; rotate on any doubt.
- You own the pipes, not the science. If a fix needs referee/model/physics changes, open an issue for that area.
- Write down infra decisions with dates in `DECISIONS.md`.

## Definition of done (hackathon)
- [ ] Repo skeleton + CLAUDE.md live Tue evening.
- [ ] CI: tests + smoke + secret scan + leakage guard on every PR, < 5 min.
- [ ] `main` protected; demo auto-deploys.
- [ ] Locked-test commit tagged; reruns logged.
- [ ] Clean-machine reproduction passes.
- [ ] Public release + working links ready before 6 p.m. ET Thursday.
