"""Reality engine tests (plan-reality-engine.md §7)."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tellurion_sim import integrate
from tellurion_sim.__main__ import main
from tellurion_sim.check import small_angle_period
from tellurion_sim.dataset import (
    RunSpec,
    generate,
    load_config,
    load_dataset,
    simulate_run,
)
from tellurion_sim.perturbations import (
    SUBTYPES,
    Perturbation,
    apply_sensor_fault,
    expected_verdict,
)
from tellurion_sim.systems import Pendulum

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs/pendulum.yaml")
SYS = Pendulum()
TIMES = integrate.time_grid(CFG["dt"], CFG["n_steps"])
P0 = SYS.nominal_params()
ONSET = 80


def run(ptype="none", subtype="none", severity="none", onset=-1, regime="gentle", seed=3, cfg=CFG):
    return simulate_run(cfg, RunSpec(0, "x", regime, ptype, subtype, severity, onset), np.random.default_rng(seed))


def zero_cfg():
    """Config with every perturbation strength set to 0."""
    cfg = {k: v for k, v in CFG.items()}
    for t in ("param_shift", "sensor_fault", "unseen_force"):
        cfg[t] = {s: {sev: 0.0 for sev in ("low", "med", "high")} for s in CFG[t]}
    # param_shift multipliers: "0 strength" means no change, i.e. x1.
    cfg["param_shift"]["length"] = {sev: 1.0 for sev in ("low", "med", "high")}
    cfg["param_shift"]["gravity"] = {sev: 1.0 for sev in ("low", "med", "high")}
    return cfg


ALL = [(t, s) for t in ("param_shift", "sensor_fault", "unseen_force") for s in SUBTYPES[t]]


# --- physics core (T12) ------------------------------------------------------

def test_energy_conservation_ideal():
    for theta0 in (0.5, 2.8):
        x, _ = integrate.simulate(SYS, np.array([theta0, 0.0]), TIMES, P0)
        e = SYS.energy(x, P0)
        assert np.abs(e - e[0]).max() / e[0] < 1e-6


def test_small_angle_period():
    expected = 2 * math.pi * math.sqrt(P0[1] / P0[0])
    assert abs(small_angle_period(SYS) - expected) / expected < 1e-3


def test_mass_invariance():
    x0 = np.array([1.0, 0.3])
    a, _ = integrate.simulate(SYS, x0, TIMES, P0)
    p = P0.copy()
    p[2] = 7.5
    b, _ = integrate.simulate(SYS, x0, TIMES, p)
    np.testing.assert_array_equal(a, b)


def test_damping_energy_non_increasing():
    p = P0.copy()
    p[3] = 0.2
    x, _ = integrate.simulate(SYS, np.array([1.0, 0.0]), TIMES, p)
    assert np.all(np.diff(SYS.energy(x, p)) <= 1e-9)


def test_positions_convention():
    pos = SYS.positions(np.array([0.0, 0.0]), P0)
    assert pos["bob_x"] == 0.0 and pos["bob_y"] == -1.0
    pos = SYS.positions(np.array([math.pi / 2, 0.0]), P0)
    assert pos["bob_x"] == pytest.approx(1.0)  # positive theta = counter-clockwise = +x


# --- regimes (T13) -----------------------------------------------------------

def test_regimes():
    rng = np.random.default_rng(0)
    g, L, m = 9.81, 1.0, 1.0
    for _ in range(200):
        th, om = SYS.sample_initial_state(rng, "gentle")
        assert abs(th) <= 0.6 and abs(om) <= 0.8
        th, om = SYS.sample_initial_state(rng, "large")
        assert 1.5 <= abs(th) <= 2.8
        assert SYS.energy(np.array([th, om]), P0) < 2 * m * g * L  # swings, doesn't go over
        x = SYS.sample_initial_state(rng, "rotating")
        assert SYS.energy(x, P0) > 2 * m * g * L
    with pytest.raises(ValueError):
        SYS.sample_initial_state(rng, "upside_down")


def test_rotating_goes_over_the_top():
    r = run(regime="rotating")
    theta = r["true_states"][:, 0]
    assert np.abs(theta).max() > math.pi


# --- perturbations (T14–T16) -------------------------------------------------

@pytest.mark.parametrize("ptype,subtype", ALL)
def test_zero_strength_is_clean(ptype, subtype):
    clean = run()
    pert = run(ptype, subtype, "high", ONSET, cfg=zero_cfg())
    for k in ("true_states", "measured_states", "measured_mask", "true_energy"):
        np.testing.assert_array_equal(clean[k], pert[k], err_msg=k)


@pytest.mark.parametrize("ptype,subtype", ALL)
def test_before_onset_matches_clean(ptype, subtype):
    clean = run()
    pert = run(ptype, subtype, "high", ONSET)
    for k in ("true_states", "measured_states", "measured_mask"):
        np.testing.assert_array_equal(clean[k][:ONSET], pert[k][:ONSET], err_msg=k)
    # ...and the perturbation actually does something after onset.
    changed = not np.array_equal(clean["true_states"], pert["true_states"]) or \
        not np.array_equal(clean["measured_states"], pert["measured_states"], equal_nan=True)
    assert changed


@pytest.mark.parametrize("subtype", SUBTYPES["param_shift"])
def test_param_shift_params(subtype):
    r = run("param_shift", subtype, "med", ONSET)
    i = {"length": 1, "gravity": 0, "damping": 3}[subtype]
    assert r["actual_params"][i] != r["nominal_params"][i]
    np.testing.assert_array_equal(np.delete(r["actual_params"], i), np.delete(r["nominal_params"], i))


@pytest.mark.parametrize("subtype", SUBTYPES["sensor_fault"])
def test_sensor_fault_never_touches_truth(subtype):
    clean = run()
    pert = run("sensor_fault", subtype, "high", ONSET)
    np.testing.assert_array_equal(clean["true_states"], pert["true_states"])
    np.testing.assert_array_equal(clean["measured_states"][:, 1], pert["measured_states"][:, 1])  # omega


def test_dropout_mask_exact():
    true = np.zeros((201, 2))
    pert = Perturbation("sensor_fault", "dropout", "high", 50, 0.5)
    meas, mask = apply_sensor_fault(true, pert, np.random.default_rng(0))
    assert np.array_equal(np.isnan(meas), mask == 0)
    assert np.all(mask[:50] == 1) and np.all(mask[:, 1] == 1)
    assert 0.3 < (mask[50:, 0] == 0).mean() < 0.7


def test_stuck_freezes_at_onset():
    r = run("sensor_fault", "stuck", "high", ONSET)
    assert np.all(r["measured_states"][ONSET:, 0] == r["true_states"][ONSET, 0])


def test_impulse_splits_at_event():
    clean = run()
    r = run("unseen_force", "impulse", "med", ONSET)
    jump = r["true_states"][ONSET, 1] - clean["true_states"][ONSET, 1]
    assert jump == pytest.approx(CFG["unseen_force"]["impulse"]["med"], rel=1e-12)  # m = L = 1
    np.testing.assert_array_equal(r["true_states"][ONSET, 0], clean["true_states"][ONSET, 0])


# --- labels, splits, dataset (T17) ------------------------------------------

def test_expected_verdict_rule():
    assert expected_verdict("gentle", "none") == "trust"
    assert expected_verdict("large", "none") == "model_error"
    assert expected_verdict("rotating", "none") == "model_error"
    assert expected_verdict("large", "sensor_fault") == "instrument_fault"
    assert expected_verdict("gentle", "param_shift") == "model_error"
    assert expected_verdict("rotating", "unseen_force") == "discovery"


def test_onset_validation():
    run("sensor_fault", "bias", "med", 0)  # 0 = from the start, allowed
    for bad in (-1, -5, len(TIMES)):
        with pytest.raises(ValueError, match="onset_step"):
            run("sensor_fault", "bias", "med", bad)
    with pytest.raises(ValueError, match="onset_step"):
        run(onset=10)  # none must use -1


# > 100 trajectories so generate() submits several chunks to the worker pool.
SMALL = {**CFG, "n_steps": 40, "splits": {"train": 60, "val": 20, "cal": 40, "test": 40},
         "onset": {"min": 5, "max": 20}}
N_SMALL = 160


@pytest.fixture(scope="module")
def small_ds():
    return generate(SMALL, seed=0, workers=1)


@pytest.fixture(scope="module")
def saved(small_ds, tmp_path_factory):
    from tellurion_sim.dataset import metadata, save_dataset
    out = tmp_path_factory.mktemp("ds") / "d"
    save_dataset(small_ds, metadata(SMALL, 0), out)
    return out


def test_determinism(small_ds):
    again = generate(SMALL, seed=0, workers=3)
    for k, v in small_ds.items():
        np.testing.assert_array_equal(v, again[k], err_msg=k)
    other = generate(SMALL, seed=1, workers=1)
    assert not np.array_equal(small_ds["true_states"], other["true_states"])


def test_schema_and_splits(small_ds):
    from tellurion_sim.check import schema_errors
    from tellurion_sim.dataset import metadata
    assert schema_errors(small_ds, metadata(SMALL, 0)) == []
    assert small_ds["true_states"].shape == (N_SMALL, 41, 2)
    assert len(np.unique(small_ds["trajectory_id"])) == N_SMALL
    tv = np.isin(small_ds["split"], ["train", "val"])
    assert set(small_ds["regime"][tv]) == {"gentle"} and set(small_ds["perturbation_type"][tv]) == {"none"}
    cal = small_ds["split"] == "cal"
    assert set(small_ds["perturbation_type"][cal]) == {"none", "param_shift", "sensor_fault", "unseen_force"}
    assert set(small_ds["regime"][cal]) == {"gentle", "large", "rotating"}


def test_train_val_load_never_opens_test(saved, monkeypatch):
    from tellurion_sim import dataset
    opened = []
    real_load = np.load
    monkeypatch.setattr(dataset.np, "load", lambda f, **kw: (opened.append(Path(f).name), real_load(f, **kw))[1])

    ds = load_dataset(saved, splits=("train", "val"))
    assert set(ds["split"]) == {"train", "val"} and len(ds["times"]) == 41
    assert opened == ["d.npz"]

    ds = load_dataset(saved)  # default: train/val/cal
    assert "test" not in set(ds["split"]) and "d.test.npz" not in opened

    ds = load_dataset(saved, splits=("test",))
    assert set(ds["split"]) == {"test"} and opened[-1] == "d.test.npz"
    assert len(load_dataset(saved, splits=("train", "val", "cal", "test"))["trajectory_id"]) == N_SMALL


def test_check_rejects_malformed(saved, tmp_path):
    from tellurion_sim.check import check_dataset
    ok, _ = check_dataset(saved)
    assert ok
    good = load_dataset(saved)
    meta = json.loads(saved.with_name("d.metadata.json").read_text())
    breakages = {
        "one state channel": lambda d: {**d, **{k: d[k][..., :1] for k in ("true_states", "measured_states", "measured_mask")}},
        "short times": lambda d: {**d, "times": d["times"][:-1]},
        "float64 states": lambda d: {**d, "true_states": d["true_states"].astype(np.float64)},
        "missing field": lambda d: {k: v for k, v in d.items() if k != "onset_step"},
    }
    for name, breakit in breakages.items():
        out = tmp_path / name.replace(" ", "_")
        # Bypass the validating writer to emulate an externally corrupted archive.
        np.savez_compressed(out.with_suffix(".npz"), **breakit(good))
        out.with_suffix(".metadata.json").write_text(json.dumps(meta))
        ok, lines = check_dataset(out)
        assert not ok, name
        assert any(line.startswith("[FAIL] contract") for line in lines), name


# --- CLI (T18) ---------------------------------------------------------------

def test_cli_end_to_end(tmp_path):
    import yaml
    cfg_path = tmp_path / "small.yaml"
    cfg_path.write_text(yaml.safe_dump(SMALL))
    out = tmp_path / "ds"
    assert main(["generate", "--config", str(cfg_path), "--seed", "0", "--out", str(out), "--workers", "1"]) == 0
    assert {p.name for p in tmp_path.iterdir()} >= {"ds.npz", "ds.test.npz", "ds.metadata.json"}
    assert main(["check", str(out)]) == 0
    assert main(["check", str(out), "--include-test"]) == 0
    csv = tmp_path / "bias.csv"
    assert main(["scenario", "--regime", "gentle", "--perturb", "sensor_fault", "--subtype", "bias",
                 "--severity", "med", "--onset", "80", "--seed", "3", "--csv", str(csv)]) == 0
    lines = csv.read_text().splitlines()
    assert lines[0].startswith("time_s,true_angle_rad,measured_angle_rad") and len(lines) == 202
    assert lines[1].endswith("sensor_fault,bias,med,80,instrument_fault")


@pytest.mark.parametrize("onset", ["-1", "201", "500"])
def test_cli_rejects_bad_onset(tmp_path, onset, capsys):
    csv = tmp_path / "x.csv"
    rc = main(["scenario", "--perturb", "sensor_fault", "--subtype", "bias", "--onset", onset, "--csv", str(csv)])
    assert rc != 0 and not csv.exists()
    assert "onset_step" in capsys.readouterr().err


def test_cli_fails_nonzero(tmp_path):
    assert main(["check", str(tmp_path / "missing")]) != 0
    assert main(["scenario", "--perturb", "sensor_fault", "--subtype", "nope", "--csv", str(tmp_path / "x.csv")]) != 0
