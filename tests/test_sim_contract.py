"""Contract checks against real simulator output, never the locked evaluation data."""

from copy import deepcopy
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from tellurion_sim.contract import LOCKED_SPLIT, dataset_errors, validate_dataset
from tellurion_sim.dataset import (
    generate,
    load_config,
    load_dataset,
    metadata,
    save_dataset,
)


@pytest.fixture(scope="module")
def generated():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/pendulum.yaml")
    cfg.update(
        n_steps=12,
        splits={"train": 2, "val": 1, "cal": 40, "test": 0},
        onset={"min": 3, "max": 7},
    )
    return generate(cfg, seed=7, workers=1), metadata(cfg, seed=7)


def test_real_engine_matches_contract(generated):
    ds, meta = generated
    assert dataset_errors(ds, meta) == []
    assert set(ds["perturbation_type"]) == {
        "none",
        "param_shift",
        "sensor_fault",
        "unseen_force",
    }


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("trajectory_id", 0, "unique"),
        ("split", "unknown", "split"),
        ("regime", "unknown", "regime"),
        ("perturbation_type", "unknown", "perturbation_type"),
        ("perturbation_subtype", "magnet", "perturbation_subtype"),
        ("severity", "high", "severity"),
        ("onset_step", 50, "onset_step"),
        ("expected_verdict", "discovery", "expected_verdict"),
    ],
)
def test_rejects_invalid_labels(generated, field, value, message):
    ds, meta = generated
    broken = deepcopy(ds)
    # The first trajectory is clean training data.
    if field == "trajectory_id":
        broken[field][1] = value
    else:
        broken[field] = broken[field].astype("U32") if broken[field].dtype.kind == "U" else broken[field]
        broken[field][0] = value
    with pytest.raises(ValueError, match=message):
        validate_dataset(broken, meta)


def test_rejects_fault_in_training_split(generated):
    ds, meta = generated
    broken = deepcopy(ds)
    i = np.flatnonzero(broken["perturbation_type"] == "sensor_fault")[0]
    broken["split"][i] = "train"
    assert any("train/val" in error for error in dataset_errors(broken, meta))


@pytest.mark.parametrize("channel", [0, 1])
def test_rejects_sensor_corruption_outside_fault(generated, channel):
    ds, meta = generated
    broken = deepcopy(ds)
    broken["measured_states"][0, 0, channel] += 1
    assert any("measured" in error for error in dataset_errors(broken, meta))


def test_rejects_omega_fault_and_pre_onset_fault(generated):
    ds, meta = generated
    i = np.flatnonzero(ds["perturbation_type"] == "sensor_fault")[0]
    for step, channel, message in [(8, 1, "omega"), (0, 0, "onset")]:
        broken = deepcopy(ds)
        broken["measured_states"][i, step, channel] += 1
        assert any(message in error for error in dataset_errors(broken, meta))


@pytest.mark.parametrize("mask,value", [(2, 0.0), (0, 0.0), (1, np.nan), (1, np.inf)])
def test_rejects_invalid_measurements(generated, mask, value):
    ds, meta = generated
    broken = deepcopy(ds)
    broken["measured_mask"][0, 0, 0] = mask
    broken["measured_states"][0, 0, 0] = value
    assert dataset_errors(broken, meta)


@pytest.mark.parametrize("field", ["times", "true_states", "trajectory_id"])
def test_bad_shapes_report_errors_without_crashing(generated, field):
    ds, meta = generated
    broken = {**ds, field: np.array(0, dtype=ds[field].dtype)}
    assert dataset_errors(broken, meta)


def test_rejects_metadata_and_time_grid_mismatch(generated):
    ds, meta = generated
    assert dataset_errors(ds, {**meta, "dt": 0.1})
    assert dataset_errors(ds, {**meta, "state_names": ["theta"]})
    broken = deepcopy(ds)
    broken["times"][4] = broken["times"][3]
    assert dataset_errors(broken, meta)


def test_ignores_extra_fields_but_rejects_non_array_required_values(generated):
    ds, meta = generated
    assert dataset_errors({**ds, "dt": meta["dt"]}, meta) == []
    assert dataset_errors({**ds, "times": ds["times"].tolist()}, meta)


@pytest.mark.parametrize("field", ["true_states", "true_energy", "nominal_params", "actual_params"])
def test_rejects_nonfinite_ground_truth(generated, field):
    ds, meta = generated
    broken = deepcopy(ds)
    broken[field].flat[0] = np.nan
    assert any(field in error for error in dataset_errors(broken, meta))


def test_writer_rejects_invalid_data_before_creating_files(generated, tmp_path):
    ds, meta = generated
    broken = deepcopy(ds)
    broken["trajectory_id"][1] = broken["trajectory_id"][0]
    with pytest.raises(ValueError, match="trajectory_id"):
        save_dataset(broken, meta, tmp_path / "invalid")
    assert not list(tmp_path.iterdir())


def test_loader_rejects_misfiled_test_rows(generated, tmp_path):
    ds, meta = generated
    save_dataset(ds, meta, tmp_path / "invalid")
    broken = deepcopy(ds)
    broken["split"][0] = "test"
    # Deliberately corrupt a development archive, using synthetic rows only.
    np.savez(tmp_path / "invalid.npz", **broken)
    with pytest.raises(ValueError, match="test"):
        load_dataset(tmp_path / "invalid")


def test_additive_archive_arrays_do_not_break_core_readers(generated, tmp_path):
    ds, meta = generated
    prefix = tmp_path / "extended"
    save_dataset(ds, meta, prefix)
    # An unrecognized object array must not trigger pickle loading either.
    np.savez(
        prefix.with_suffix(".npz"),
        **ds,
        true_positions=ds["true_states"].copy(),
        extension=np.array([object()], dtype=object),
    )
    actual = load_dataset(prefix)
    for key in ds:
        np.testing.assert_array_equal(actual[key], ds[key], err_msg=key)


def test_writer_preserves_trajectory_aligned_extensions(generated, tmp_path):
    ds, meta = generated
    extended = {**ds, "true_positions": ds["true_states"].copy()}
    assert dataset_errors(extended, meta) == []
    save_dataset(extended, meta, tmp_path / "extended_writer")
    with np.load(tmp_path / "extended_writer.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["true_positions"], extended["true_positions"])
    assert set(load_dataset(tmp_path / "extended_writer")) == set(ds)


def test_writer_rejects_unpartitionable_extension_before_writing(generated, tmp_path):
    ds, meta = generated
    with pytest.raises(ValueError, match="extension arrays"):
        save_dataset({**ds, "scalar_setting": np.array(0.05)}, meta, tmp_path / "invalid_extension")
    assert not list(tmp_path.iterdir())


def test_system_hook_controls_system_specific_contract_rules(generated, monkeypatch):
    from tellurion_sim.systems import SYSTEMS
    from tellurion_sim.systems.base import System

    class TwoAngleSystem(System):
        name = "two_angles_fixture"
        state_names: ClassVar[list[str]] = ["theta1", "theta2", "omega1", "omega2"]
        param_names: ClassVar[list[str]] = ["signed_coupling", "g", "L1", "L2", "m1", "m2"]

    monkeypatch.setitem(SYSTEMS, TwoAngleSystem.name, TwoAngleSystem)
    ds, meta = generated
    ds = deepcopy(ds)
    for key in ("true_states", "measured_states", "measured_mask"):
        ds[key] = np.concatenate([ds[key], ds[key]], axis=2)
    params = np.tile(np.array([-1, 9.81, 1, 1, 1, 1], dtype=np.float32), (len(ds["split"]), 1))
    ds["nominal_params"], ds["actual_params"] = params, params.copy()
    fault = np.flatnonzero(ds["perturbation_type"] == "sensor_fault")[0]
    ds["measured_states"][fault, ds["onset_step"][fault] :, 1] += 0.1
    meta = {
        **meta,
        "system": TwoAngleSystem.name,
        "state_names": TwoAngleSystem.state_names,
        "param_names": TwoAngleSystem.param_names,
    }
    assert dataset_errors(ds, meta) == []

    monkeypatch.setattr(TwoAngleSystem, "dataset_errors", lambda self, data: ["custom system rule"])
    assert "custom system rule" in dataset_errors(ds, meta)


@pytest.mark.parametrize("field,index,value", [("nominal_params", 0, -1), ("actual_params", 3, -1)])
def test_pendulum_hook_still_enforces_physical_parameters(generated, field, index, value):
    ds, meta = generated
    broken = deepcopy(ds)
    broken[field][0, index] = value
    assert any(field in error for error in dataset_errors(broken, meta))


def test_loader_requires_metadata_and_checker_reports_missing_sidecar(generated, tmp_path):
    from tellurion_sim.check import check_dataset

    ds, _ = generated
    prefix = tmp_path / "missing_metadata"
    np.savez(prefix.with_suffix(".npz"), **ds)
    with pytest.raises((ValueError, FileNotFoundError), match="metadata"):
        load_dataset(prefix)
    ok, lines = check_dataset(prefix)
    assert not ok
    assert any("[FAIL]" in line and "metadata" in line for line in lines)


@pytest.mark.parametrize(
    "change",
    [
        {"n_steps": None},
        {"n_steps": True},
        {"dt": "0.05"},
        {"dt": float("nan")},
        {"system": []},
        {"state_names": None},
        {"param_names": ["L", "g", "m", "b"]},
    ],
)
def test_loader_and_checker_reject_malformed_metadata(generated, tmp_path, change):
    import json

    from tellurion_sim.check import check_dataset

    ds, meta = generated
    prefix = tmp_path / "bad_metadata"
    save_dataset(ds, meta, prefix)
    prefix.with_suffix(".metadata.json").write_text(json.dumps({**meta, **change}), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        load_dataset(prefix)
    ok, lines = check_dataset(prefix)
    assert not ok
    assert any(line.startswith("[FAIL] metadata") for line in lines)


def test_unreadable_synthetic_test_archive_retains_development_diagnostics(generated, tmp_path):
    from tellurion_sim.check import check_dataset

    ds, meta = generated
    prefix = tmp_path / "unreadable"
    save_dataset(ds, meta, prefix)
    (tmp_path / "unreadable.test.npz").write_bytes(b"invalid archive")
    with pytest.raises(ValueError, match="cannot read archive"):
        load_dataset(prefix, splits=(LOCKED_SPLIT,))
    ok, lines = check_dataset(prefix, include_test=True)
    assert not ok
    assert any(f"seed={meta['seed']}" in line and meta["git_commit"][:12] in line for line in lines)
    assert any("label counts per split" in line for line in lines)
    assert any("archive could not be read" in line for line in lines)


def test_loader_validates_each_archive_once(generated, tmp_path, monkeypatch):
    from tellurion_sim import contract

    ds, meta = generated
    prefix = tmp_path / "once"
    save_dataset(ds, meta, prefix)
    calls = []
    original = contract.dataset_errors

    def track(data, metadata=None):
        calls.append(len(data["trajectory_id"]))
        return original(data, metadata)

    monkeypatch.setattr(contract, "dataset_errors", track)
    load_dataset(prefix)
    assert calls == [len(ds["trajectory_id"])]
    calls.clear()
    load_dataset(prefix, splits=("train", "val", "cal", "test"))
    assert calls == [len(ds["trajectory_id"]), 0]


@pytest.mark.parametrize("damage", ["duplicate_id", "time_grid"])
def test_loader_rejects_cross_archive_inconsistencies(generated, tmp_path, damage):
    ds, meta = generated
    prefix = tmp_path / "cross_archive"
    save_dataset(ds, meta, prefix)
    synthetic = {key: value.copy() if key == "times" else value[:1].copy() for key, value in ds.items()}
    synthetic["split"] = np.full(1, LOCKED_SPLIT)
    synthetic["trajectory_id"] = np.array([1000], dtype=np.int64)
    if damage == "duplicate_id":
        synthetic["trajectory_id"][0] = ds["trajectory_id"][0]
    else:
        # Each archive is valid within grid tolerance, but the grids must agree exactly.
        synthetic["times"][4] += 1e-14
    np.savez(tmp_path / "cross_archive.test.npz", **synthetic)
    with pytest.raises(ValueError, match="trajectory_id|time|grid"):
        load_dataset(prefix, splits=("train", "val", "cal", "test"))


@pytest.mark.parametrize("damage", ["missing_field", "bad_dtype", "duplicate_id", "bad_label"])
def test_corrupt_synthetic_test_archive_is_rejected(generated, tmp_path, damage):
    from tellurion_sim.check import check_dataset

    ds, meta = generated
    prefix = tmp_path / "corrupt"
    save_dataset(ds, meta, prefix)
    synthetic = {key: value.copy() if key == "times" else value[:2].copy() for key, value in ds.items()}
    synthetic["split"] = np.array(["test", "test"])
    synthetic["trajectory_id"] += 1000
    if damage == "missing_field":
        del synthetic["onset_step"]
    elif damage == "bad_dtype":
        synthetic["true_states"] = synthetic["true_states"].astype(np.float64)
    elif damage == "duplicate_id":
        synthetic["trajectory_id"][1] = synthetic["trajectory_id"][0]
    else:
        synthetic["expected_verdict"] = np.array(["discovery", "discovery"])
    np.savez(tmp_path / "corrupt.test.npz", **synthetic)
    # The corrupt synthetic test archive must not affect development loading.
    load_dataset(prefix)
    with pytest.raises(ValueError):
        load_dataset(prefix, splits=("test",))
    ok, lines = check_dataset(prefix, include_test=True)
    assert not ok
    assert any("[FAIL] contract" in line for line in lines)
    assert any(f"seed={meta['seed']}" in line and meta["git_commit"][:12] in line for line in lines)
    assert any("label counts per split" in line for line in lines)


def test_checker_reports_all_contract_errors_with_provenance(generated, tmp_path):
    from tellurion_sim.check import check_dataset

    ds, meta = generated
    prefix = tmp_path / "multiple_errors"
    save_dataset(ds, meta, prefix)
    broken = deepcopy(ds)
    broken["trajectory_id"][1] = broken["trajectory_id"][0]
    broken["measured_mask"][0, 0, 0] = 2
    np.savez(prefix.with_suffix(".npz"), **broken)
    ok, lines = check_dataset(prefix)
    assert not ok
    assert any("[FAIL] contract" in line and "trajectory_id" in line for line in lines)
    assert any("[FAIL] contract" in line and "measured_mask" in line for line in lines)
    assert any(f"seed={meta['seed']}" in line and meta["git_commit"][:12] in line for line in lines)
    assert any("label counts per split" in line for line in lines)
    assert not any(line.startswith("[PASS]") for line in lines)


def test_round_trip_and_default_loading_never_opens_locked_file(generated, tmp_path, monkeypatch):
    ds, meta = generated
    prefix = tmp_path / "pendulum"
    save_dataset(ds, meta, prefix)
    # Even an unreadable locked file must not affect development loading.
    (tmp_path / "pendulum.test.npz").write_bytes(b"locked: do not open")
    real_load = np.load
    opened = []

    def tracked_load(path, **kwargs):
        opened.append(Path(path).name)
        return real_load(path, **kwargs)

    monkeypatch.setattr(np, "load", tracked_load)
    actual = load_dataset(prefix, splits=("train", "val", "cal"))
    for key in ds:
        np.testing.assert_array_equal(ds[key], actual[key], err_msg=key)
    assert opened == ["pendulum.npz"]


def test_empty_split_archive_is_valid(generated, tmp_path):
    ds, meta = generated
    empty = {key: value if key == "times" else value[:0] for key, value in ds.items()}
    validate_dataset(empty, meta)
    save_dataset(empty, meta, tmp_path / "empty")
    actual = load_dataset(tmp_path / "empty")
    assert actual["true_states"].shape == (0, len(ds["times"]), 2)


def test_checker_handles_no_ideal_trajectories(generated, tmp_path):
    from tellurion_sim.check import check_dataset

    ds, meta = generated
    keep = ds["perturbation_type"] == "unseen_force"
    forced = {key: value if key == "times" else value[keep] for key, value in ds.items()}
    save_dataset(forced, meta, tmp_path / "forced")
    ok, lines = check_dataset(tmp_path / "forced")
    assert ok
    assert any(line.startswith("[SKIP] energy drift") for line in lines)


@pytest.mark.parametrize("ptype,subtype", [("sensor_fault", "dropout"), ("param_shift", "length")])
def test_csv_matches_sensor_and_parameter_timing(generated, tmp_path, ptype, subtype):
    import csv

    from tellurion_sim.dataset import RunSpec, simulate_run
    from tellurion_sim.export import COLUMNS, write_run_csv
    from tellurion_sim.systems import Pendulum

    ds, _ = generated
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/pendulum.yaml")
    cfg["n_steps"] = len(ds["times"]) - 1
    onset = 4
    if ptype == "sensor_fault":
        # Force dropout to exercise missing CSV cells deterministically.
        cfg[ptype][subtype]["high"] = 1.0
    run = simulate_run(
        cfg,
        RunSpec(0, "scenario", "gentle", ptype, subtype, "high", onset),
        np.random.default_rng(0),
    )
    labels = {
        "perturbation_type": ptype,
        "perturbation_subtype": subtype,
        "severity": "high",
        "onset_step": onset,
        "expected_verdict": run["expected_verdict"],
    }
    path = write_run_csv(tmp_path / "scenario.csv", Pendulum(), ds["times"], run, labels)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == COLUMNS
        rows = list(reader)
    assert len(rows) == len(ds["times"])
    for k, row in enumerate(rows):
        assert all(row[key] == "" for key in ("predicted_angle_rad", "confidence_0to1", "pred_x", "pred_y"))
        params = run["nominal_params"] if k < onset else run["actual_params"]
        assert float(row["true_x"]) == pytest.approx(params[1] * np.sin(run["true_states"][k, 0]), abs=1e-8)
        assert float(row["true_y"]) == pytest.approx(-params[1] * np.cos(run["true_states"][k, 0]), abs=1e-8)
        if ptype == "sensor_fault":
            assert row["measured_valid"] == ("1" if k < onset else "0")
            assert (row["measured_angle_rad"] == "") == (k >= onset)
        assert row["expected_verdict"] == run["expected_verdict"]


def test_real_calibration_data_can_be_consumed_by_referee(generated):
    from referee.model import StubForecaster
    from referee.referee import BANNED_KEYS, Referee

    ds, meta = generated
    i = np.flatnonzero((ds["split"] == "cal") & (ds["expected_verdict"] == "trust"))[0]
    theta, omega = ds["measured_states"][i].T
    params = dict(zip(meta["param_names"], map(float, ds["nominal_params"][i])))
    forecast_theta, forecast_omega = StubForecaster(**params).forecast_last_step(theta, omega, meta["dt"])
    history = {
        "dt": meta["dt"],
        "measured_theta": theta,
        "measured_omega": omega,
        "measured_mask": ds["measured_mask"][i, :, 0],
        "nominal_params": params,
        "forecast_theta": forecast_theta,
        "forecast_omega": forecast_omega,
    }
    assert not BANNED_KEYS.intersection(history)
    referee = Referee()
    referee.fit_thresholds([history])
    assert referee.step(history)["verdict"] == "trust"
