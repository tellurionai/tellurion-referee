# One command per step. Targets print a notice until their ticket lands; replace the echo with the real command.
CONFIG ?= configs/base.yaml
PY ?= python
# the shared frozen model; override to test a local one: make eval-cal MODEL=artifacts/world_model
MODEL ?= models/world_model

.PHONY: install data train publish-model eval-cal eval-test replays figures test smoke lint

install:
	$(PY) -m pip install -e '.[dev]'

data:
	$(PY) -m tellurion_sim generate --system pendulum --config configs/pendulum.yaml --seed 0 --out data/pendulum_v1
	$(PY) -m tellurion_sim check data/pendulum_v1

train:
	$(PY) scripts/train.py --data data/pendulum_v1.npz --out artifacts/world_model

# Copy a trained model into models/world_model, where it is committed and shared.
# Refuses to overwrite: replacing the shared model needs a deliberate, reviewed PR.
publish-model:
	@test -f artifacts/world_model/manifest.json || (echo "no model in artifacts/world_model: run make train" && exit 1)
	@test ! -f models/world_model/manifest.json || (echo "models/world_model already has a model; remove it in a reviewed PR first" && exit 1)
	cp artifacts/world_model/manifest.json artifacts/world_model/member_*.pt models/world_model/
	$(PY) -c "from referee.model import verify_checkpoints; verify_checkpoints('models/world_model'); print('checkpoints verified')"

eval-cal:
	$(PY) scripts/evaluate.py --config $(CONFIG) --split cal --model $(MODEL)

# The locked test split is evaluated ONCE, at Gate 3 (Thu 12:00 ET). Do not run this before then.
eval-test:
	$(PY) scripts/evaluate.py --config $(CONFIG) --split test --confirm-locked-test --model $(MODEL)

# Which results replays/figures use: cal during development, test after Gate 3
SPLIT ?= test
replays:
	$(PY) scripts/export_replays.py --split $(SPLIT)

figures:
	$(PY) scripts/make_figures.py --input results/$(SPLIT)_metrics.json

test:
	$(PY) -m pytest -q

# End-to-end on the tiny config, < 5 minutes on CPU. Becomes: data + train + eval-cal with configs/smoke.yaml
smoke:
	@echo "smoke: stub until data/train/eval-cal exist (uses configs/smoke.yaml)"

lint:
	ruff check .
