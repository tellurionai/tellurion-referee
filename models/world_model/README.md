# The shared world model

This folder holds **the one frozen world model** every evaluation uses: five checkpoints
(`member_0.pt` to `member_4.pt`) and `manifest.json` with their SHA256 hashes, the
training settings and the dataset fingerprint. About 0.4 MB.

Why it is committed: training is not bit-identical across machines, so a model trained
on two laptops gives two slightly different models. The locked test (Gate 3) and every
reported number must come from the same files, and anyone must be able to rerun them.

- `make eval-cal` and `make eval-test` use this folder by default.
- `load_ensemble` checks every SHA256 before loading; a changed file fails loudly.
- `results/model_report.json` must list the same `checkpoint_sha256` as `manifest.json`.

## Adding the model (once)
```bash
make train            # or keep the model you already trained in artifacts/world_model
make publish-model    # copies it here and verifies the hashes
git add models/world_model
```
If you trained a new model rather than the one in `results/model_report.json`, also
regenerate the report so the hashes match:
`python scripts/train.py --data data/pendulum_v1.npz --out models/world_model --report-only`

## Changing it
Don't, after Gate 3. Before that, replace it only in a reviewed PR (this folder needs a
maintainer's approval) and regenerate `results/model_report.json` in the same PR.
