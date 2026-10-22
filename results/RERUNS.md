# Locked test reruns

The locked test split is evaluated once, at Gate 3. Any later rerun is logged here
automatically by `scripts/evaluate.py --split test --rerun-reason "..."`, with its date,
commit and reason.

- 2026-10-22T17:59:54Z: Gate 3, the single locked-test evaluation, commit edc0d7a
  (tag `v0.1-hackathon-test`), model 3e2118e1, dataset pendulum_v1.test.npz (sha256 016f1a36…).
  No reruns.
