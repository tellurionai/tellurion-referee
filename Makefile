# One command per step. Targets print a notice until their ticket lands; replace the echo with the real command.
CONFIG ?= configs/base.yaml
PY ?= python

.PHONY: install data train eval-cal eval-test figures test smoke lint

install:
	$(PY) -m pip install -e '.[dev]'

data:
	$(PY) -m tellurion_sim generate --system pendulum --config configs/pendulum.yaml --seed 0 --out data/pendulum_v1
	$(PY) -m tellurion_sim check data/pendulum_v1

train:
	$(PY) scripts/train.py --data data/pendulum_v1.npz --out artifacts/world_model

eval-cal:
	$(PY) scripts/evaluate.py --config $(CONFIG) --split cal

# The locked test split is evaluated ONCE, at Gate 3 (Thu 12:00 ET). Do not run this before then.
eval-test:
	$(PY) scripts/evaluate.py --config $(CONFIG) --split test --confirm-locked-test

figures:
	@echo "figures: not implemented yet (docs & figures). Will run: $(PY) scripts/make_figures.py --input results/test_metrics.json"

test:
	$(PY) -m pytest -q

# End-to-end on the tiny config, < 5 minutes on CPU. Becomes: data + train + eval-cal with configs/smoke.yaml
smoke:
	@echo "smoke: stub until data/train/eval-cal exist (uses configs/smoke.yaml)"

lint:
	ruff check .
