"""World-model acceptance tests use synthetic fixtures, never the locked dataset."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from referee.model import (
    DynamicsModel, Ensemble, TrainConfig, load_split, train_ensemble,
    save_ensemble, load_ensemble, model_report,
)
from referee.physics_fit import simulate


@pytest.fixture
def handoff(tmp_path):
    # Disjoint trajectories, including clean OOD calibration examples.
    splits = ["train"] * 8 + ["val"] * 3 + ["cal"] * 3 + ["test"]
    regimes = ["gentle"] * 12 + ["large", "rotating", "gentle"]
    starts = [(t, 0.2) for t in np.linspace(-0.6, 0.6, 8)]
    starts += [(0.15, -0.1), (-0.35, 0.3), (0.5, -0.2)]
    starts += [(0.25, 0.0), (2.0, 0.0), (0.1, 8.0), (0.0, 0.0)]
    states = np.array([np.stack(simulate(th, om, 0.05, 40, 9.81, 1.0, 0.0), -1)
                       for th, om in starts], dtype=np.float32)
    states[-1] = np.nan  # Poison the locked fixture row; it must never be selected.
    arrays = dict(times=np.arange(41, dtype=np.float64) * 0.05,
                  measured_states=states, measured_mask=np.ones_like(states, dtype=np.uint8),
                  trajectory_id=np.arange(len(states), dtype=np.int64),
                  split=np.array(splits), regime=np.array(regimes),
                  perturbation_type=np.full(len(states), "none"))
    path = tmp_path / "handoff.npz"
    np.savez(path, **arrays)
    path.with_suffix(".metadata.json").write_text(json.dumps({
        "system": "pendulum", "state_names": ["theta", "omega"],
        "dt": 0.05, "horizon": 40, "seed": 71, "git_commit": "fixture",
    }))
    return path, arrays


def test_handoff_selection_and_locked_split(handoff):
    path, _ = handoff
    train = load_split(path, "train")
    assert train.states.shape == (8, 41, 2)
    assert np.isfinite(train.states).all()
    assert train.dt == pytest.approx(0.05)
    with pytest.raises(ValueError, match="locked"):
        load_split(path, "test")


def test_compressed_handoff_and_masked_transitions(handoff):
    path, arrays = handoff
    arrays["measured_states"][0, 2] = np.nan
    arrays["measured_mask"][0, 2] = 0
    np.savez_compressed(path, **arrays)
    train = load_split(path, "train")
    x, y = train.transitions()
    assert len(x) == 8 * 40 - 2
    assert np.isfinite(x).all() and np.isfinite(y).all()


@pytest.mark.parametrize("corruption", ["ids", "regime", "perturbation", "dt", "n_steps"])
def test_handoff_rejects_invalid_training_contract(handoff, corruption):
    path, arrays = handoff
    if corruption == "ids":
        arrays["trajectory_id"][8] = arrays["trajectory_id"][0]
    elif corruption == "regime":
        arrays["regime"][0] = "large"
    elif corruption == "perturbation":
        arrays["perturbation_type"] = np.full(len(arrays["split"]), "sensor_fault")
    elif corruption == "dt":
        arrays["times"][3] += 0.01
    else:
        metadata_path = path.with_suffix(".metadata.json")
        metadata = json.loads(metadata_path.read_text())
        metadata["n_steps"] = 99
        metadata_path.write_text(json.dumps(metadata))
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        load_split(path, "train")


def test_fortran_order_handoff_uses_numpy_reader(handoff):
    path, arrays = handoff
    arrays["measured_states"] = np.asfortranarray(arrays["measured_states"])
    np.savez_compressed(path, **arrays)
    selected = load_split(path, "train")
    np.testing.assert_array_equal(selected.states, arrays["measured_states"][:8])


@pytest.fixture(scope="module")
def fast_config():
    return TrainConfig(seeds=(0, 1, 2, 3, 4), hidden_sizes=(64, 64),
                       epochs=35, batch_size=128, patience=10, threads=1)


def test_training_determinism_normalization_and_checkpoint(handoff, fast_config, tmp_path):
    path, _ = handoff
    train, val = load_split(path, "train"), load_split(path, "val")
    ensemble, record = train_ensemble(train, val, fast_config)
    repeat, _ = train_ensemble(train, val, fast_config)
    x, _ = train.transitions()
    encoded = np.stack([np.sin(x[:, 0]), np.cos(x[:, 0]), x[:, 1]], -1)
    np.testing.assert_allclose(ensemble.models[0].input_mean.numpy(), encoded.mean(0), atol=1e-6)
    for a, b in zip(ensemble.models, repeat.models):
        for name, value in a.state_dict().items():
            assert torch.equal(value, b.state_dict()[name])
        assert not any(p.requires_grad for p in a.parameters())
    assert len(record["members"]) == 5
    assert all(m["best_epoch"] <= m["epochs_run"] for m in record["members"])
    out = tmp_path / "model"
    save_ensemble(out, ensemble, record)
    loaded = load_ensemble(out)
    expected = ensemble.forecast([[0.2, 0.0]], 10)
    actual = loaded.forecast([[0.2, 0.0]], 10)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    manifest = json.loads((out / "manifest.json").read_text())
    checkpoint = out / manifest["checkpoints"][0]["file"]
    with checkpoint.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        load_ensemble(out)


def test_autoregressive_members_and_wrap_safe_input():
    models = [DynamicsModel(hidden_sizes=(64, 64)) for _ in range(2)]
    for model, delta in zip(models, (0.1, 0.3)):
        for p in model.parameters():
            torch.nn.init.zeros_(p)
        model.network[-1].bias.data[:] = torch.tensor([delta, 0.0])
    ensemble = Ensemble(models, dt=0.05)
    mean, std = ensemble.forecast([[0.0, 0.0]], 3)
    torch.testing.assert_close(mean[0, :, 0], torch.tensor([0.2, 0.4, 0.6]))
    torch.testing.assert_close(std[0, :, 0], torch.tensor([0.1, 0.2, 0.3]))
    assert mean.shape == std.shape == (1, 3, 2)
    assert not mean.requires_grad
    x = torch.tensor([[0.2, 0.4], [0.2 + 2 * np.pi, 0.4]])
    torch.testing.assert_close(models[0](x)[0], models[0](x)[1])
    with pytest.raises(ValueError):
        ensemble.forecast([[np.nan, 0.0]], 1)
    with pytest.raises(ValueError):
        ensemble.forecast([[0.0, 0.0]], 0)
    with pytest.raises(ValueError, match="dt"):
        ensemble.forecast_step(0.1, 0.0, 0.1)


def test_angle_wrap_does_not_create_a_large_training_target(handoff):
    path, arrays = handoff
    arrays["measured_states"][0, 0, 0] = np.pi - 0.01
    arrays["measured_states"][0, 1, 0] = -np.pi + 0.01
    np.savez(path, **arrays)
    x, delta = load_split(path, "train").transitions()
    assert delta[0, 0] == pytest.approx(0.02, abs=1e-6)
    torch.manual_seed(17)
    model = DynamicsModel()
    shifted = torch.tensor([[0.2, 0.4], [0.2 + 2 * np.pi, 0.4]])
    torch.testing.assert_close(model(shifted)[0], model(shifted)[1], atol=1e-6, rtol=1e-5)


def test_early_stopping_restores_best_epoch_and_ignores_cal(handoff):
    path, arrays = handoff
    config = TrainConfig(seeds=(7,), epochs=10, patience=2, min_delta=1e9, hidden_sizes=(64, 64))
    train, val = load_split(path, "train"), load_split(path, "val")
    ensemble, record = train_ensemble(train, val, config)
    member = record["members"][0]
    assert member["best_epoch"] == 1 and member["epochs_run"] == 3
    first_epoch, _ = train_ensemble(train, val, TrainConfig(seeds=(7,), epochs=1, hidden_sizes=(64, 64)))
    for name, value in ensemble.models[0].state_dict().items():
        assert torch.equal(value, first_epoch.models[0].state_dict()[name])
    arrays["measured_states"][arrays["split"] == "cal"] = 1e6
    np.savez(path, **arrays)
    changed_cal, _ = train_ensemble(load_split(path, "train"), load_split(path, "val"), config)
    for name, value in ensemble.models[0].state_dict().items():
        assert torch.equal(value, changed_cal.models[0].state_dict()[name])


def test_report_sustained_crossing_and_censoring(handoff):
    path, arrays = handoff
    # A stationary predictor has errors .1, .3, .3, .3 then zero: crossing is step 2.
    arrays["measured_states"][11] = 0
    arrays["measured_states"][11, 1:5, 0] = [0.1, 0.3, 0.3, 0.3]
    arrays["measured_states"][12] = 0  # large regime never crosses
    # No trustworthy initial observation means no rollout horizon.
    arrays["measured_states"][13, 0] = np.nan
    arrays["measured_mask"][13, 0] = 0
    np.savez(path, **arrays)
    model = DynamicsModel()
    for p in model.parameters():
        torch.nn.init.zeros_(p)
    ensemble = Ensemble([model], dt=0.05)
    report = model_report(ensemble, load_split(path, "cal"), bootstrap_samples=10)
    gentle = report["regimes"]["gentle"]
    assert gentle["reliability_horizon_steps_median"] == 2
    assert gentle["reliability_horizon_seconds_median"] == pytest.approx(0.1)
    assert gentle["reliability_horizon_steps_ci95"] == [2.0, 2.0]
    assert gentle["n_right_censored"] == 0
    assert report["regimes"]["large"]["n_right_censored"] == 1
    assert report["regimes"]["large"]["reliability_horizon_steps_median"] == 40
    assert report["regimes"]["rotating"]["reliability_horizon_steps_median"] is None


def test_report_measures_held_out_regimes(handoff, fast_config):
    path, _ = handoff
    ensemble, _ = train_ensemble(load_split(path, "train"), load_split(path, "val"), fast_config)
    report = model_report(ensemble, load_split(path, "cal"), bootstrap_samples=50)
    assert report["split"] == "cal"
    assert report["threshold"] == 0.25 and report["sustained_steps"] == 3
    gentle = report["regimes"]["gentle"]
    assert gentle["n_trajectories"] == 1
    assert gentle["one_step_normalized_rmse"] < 0.1
    assert report["regimes"]["large"]["one_step_normalized_rmse"] > gentle["one_step_normalized_rmse"]
    assert report["regimes"]["rotating"]["n_trajectories"] == 1
    assert "reliability_horizon_steps_ci95" in gentle
    json.dumps(report, allow_nan=False)
    val_report = model_report(ensemble, load_split(path, "val"), bootstrap_samples=10)
    assert val_report["regimes"]["rotating"]["status"] == "unavailable"


def test_cli_train_and_report_only(handoff, tmp_path):
    path, _ = handoff
    root = Path(__file__).resolve().parents[1]
    out, report = tmp_path / "checkpoints", tmp_path / "report.json"
    command = [sys.executable, "-B", str(root / "scripts/train.py"), "--data", str(path),
               "--out", str(out), "--report", str(report)]
    subprocess.run(command + ["--epochs", "2", "--bootstrap-samples", "10"], check=True)
    manifest = (out / "manifest.json").read_bytes()
    subprocess.run(command + ["--report-only", "--bootstrap-samples", "10"], check=True)
    assert (out / "manifest.json").read_bytes() == manifest
    assert json.loads(report.read_text())["checkpoint_sha256"]


def test_cli_missing_train_and_yaml_config(handoff, tmp_path):
    path, arrays = handoff
    from train import main
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps({"data": str(path), "out": str(tmp_path / "weights"),
                                      "report": str(tmp_path / "result.json"),
                                      "model": {"epochs": 1, "seeds": [0]}, "bootstrap_samples": 10}))
    main(["--config", str(config_path)])
    assert (tmp_path / "weights/manifest.json").exists()
    arrays["split"][arrays["split"] == "train"] = "cal"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="no train trajectories"):
        load_split(path, "train")
    with pytest.raises(SystemExit) as exc:
        main(["--config", str(config_path), "--out", str(tmp_path / "missing")])
    assert exc.value.code == 2
    assert not (tmp_path / "missing").exists()


def _constant_ensemble(step=0.1):
    """Each forecast step adds `step` to theta, so k-step rollouts are exact."""
    model = DynamicsModel((4,))
    for p in model.parameters():
        torch.nn.init.zeros_(p)
    model.network[-1].bias.data[:] = torch.tensor([step, 0.0])
    return Ensemble([model], dt=0.05)


def test_forecast_last_step_skips_dropped_readings():
    ensemble = _constant_ensemble()
    theta = np.array([0.0, 0.5, 1.0, 1.2])
    omega = np.zeros(4)
    # Previous sample valid: one-step forecast from it, as before.
    assert ensemble.forecast_last_step(theta, omega, 0.05)[0] == pytest.approx(1.1)
    # Previous sample dropped (NaN): roll out two steps from the last valid one.
    theta[2] = np.nan
    assert ensemble.forecast_last_step(theta, omega, 0.05)[0] == pytest.approx(0.7)
    # No valid earlier reading: NaN like StubForecaster, never an exception.
    theta[:3] = np.nan
    assert np.isnan(ensemble.forecast_last_step(theta, omega, 0.05)).all()
    with pytest.raises(ValueError, match="dt"):
        ensemble.forecast_last_step(theta, omega, 0.1)


def test_portable_path_never_records_absolute_paths(tmp_path):
    from referee.model import REPO_ROOT, portable_path
    assert portable_path(REPO_ROOT / "artifacts" / "world_model" / "manifest.json") == "artifacts/world_model/manifest.json"
    assert portable_path(tmp_path / "elsewhere" / "data.npz") == "data.npz"
