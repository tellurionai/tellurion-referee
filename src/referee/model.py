"""Frozen pendulum world model and the reality-engine data handoff.

Train on clean measured train trajectories; early-stop on val. Forecasts take
only [theta, omega] states. Angles stay on the input's continuous branch while
inputs and error differences are wrap-safe. Checkpoints include all train-only
normalization buffers. The original StubForecaster remains for referee fixtures.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import logging
import platform
import time
import zipfile

import numpy as np
import torch
from scipy.stats import bootstrap
from torch import nn

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]


def portable_path(path) -> str:
    """Repo-relative POSIX path for files recorded in committed reports.

    Absolute local paths would leak machine layouts and differ per machine;
    files outside the repo are recorded by name only.
    """
    path = Path(path).resolve()
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else path.name


def angle_difference(a, b):
    """Signed shortest angular difference in radians (NumPy arrays)."""
    return np.arctan2(np.sin(a - b), np.cos(a - b))


@dataclass
class SplitData:
    states: np.ndarray
    mask: np.ndarray
    trajectory_id: np.ndarray
    regime: np.ndarray
    perturbation_type: np.ndarray
    times: np.ndarray
    split: str
    metadata: dict
    source: str

    @property
    def dt(self) -> float:
        return float(self.times[1] - self.times[0])

    def transitions(self):
        valid = self.mask.all(-1) & np.isfinite(self.states).all(-1)
        pairs = valid[:, :-1] & valid[:, 1:]
        x, following = self.states[:, :-1][pairs], self.states[:, 1:][pairs]
        if not len(x):
            raise ValueError(f"No valid adjacent transitions in {self.split}")
        delta = following - x
        delta[:, 0] = angle_difference(following[:, 0], x[:, 0])
        return x.astype(np.float32), delta.astype(np.float32)


def _read_rows(archive, name, indices, total_rows):
    """Extract one member to disk; NumPy maps it and copies only selected rows.

    NPZ files cannot be memory-mapped directly. The temporary NPY contains the
    full member, but test rows are never selected for training or reporting.
    NumPy handles the file format, dtype, and C/Fortran memory layout.
    """
    with TemporaryDirectory(prefix="referee-array-") as directory:
        path = archive.extract(f"{name}.npy", directory)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if array.ndim == 0 or len(array) != total_rows:
                raise ValueError(f"{name} must have N rows")
            return np.array(array[indices], copy=True)
        finally:
            del array  # Release the mapping before deleting the file on Windows.


def load_split(path: str | Path, split: str) -> SplitData:
    """Load train, val, or cal from contract NPZ + .metadata.json files.

    Split labels and trajectory IDs are inspected globally for leakage. State
    rows are selected before decoding; true_states/actual_params are never read.
    This training/reporting entry point deliberately cannot select locked test.
    """
    if split not in {"train", "val", "cal"}:
        raise ValueError("Only train, val, and cal are allowed; test is locked")
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if metadata.get("system") != "pendulum" or metadata.get("state_names") != ["theta", "omega"]:
        raise ValueError("Expected pendulum state_names [theta, omega]")
    with np.load(path, allow_pickle=False) as arrays:
        splits = arrays["split"].astype(str)
        ids = arrays["trajectory_id"]
        times = arrays["times"]
    if splits.ndim != 1 or ids.shape != splits.shape or ids.dtype.kind not in "iu":
        raise ValueError("split and integer trajectory_id must have shape [N]")
    if not set(splits) <= {"train", "val", "cal", "test"}:
        raise ValueError("Unknown split label")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("trajectory_id must be unique, including across splits")
    indices = np.flatnonzero(splits == split)
    if not len(indices):
        raise ValueError(f"Dataset has no {split} trajectories; provide the reality-engine train/val handoff")
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or times[1] <= times[0] or not np.allclose(np.diff(times), times[1] - times[0])):
        raise ValueError("times must be a finite, increasing, uniformly spaced grid")
    if not np.isclose(metadata["dt"], times[1] - times[0]):
        raise ValueError("metadata dt disagrees with times")
    n_steps = metadata.get("n_steps", metadata.get("horizon", len(times) - 1))
    if n_steps != len(times) - 1:
        raise ValueError("metadata horizon disagrees with times")
    with zipfile.ZipFile(path) as archive:
        selected = {key: _read_rows(archive, key, indices, len(ids)) for key in
                    ("measured_states", "measured_mask", "regime", "perturbation_type")}
    states, mask = selected["measured_states"], selected["measured_mask"]
    if states.shape != (len(indices), len(times), 2) or mask.shape != states.shape:
        raise ValueError("measured_states and measured_mask must have shape [N, T+1, 2]")
    if states.dtype.kind != "f" or not np.isin(mask, [0, 1]).all():
        raise ValueError("States must be floating point and masks binary")
    if not np.isfinite(states[mask.astype(bool)]).all():
        raise ValueError("Valid sensor readings must be finite; dropped readings need mask=0")
    regimes = selected["regime"].astype(str)
    perturbations = selected["perturbation_type"].astype(str)
    if regimes.shape != (len(indices),) or perturbations.shape != regimes.shape:
        raise ValueError("regime and perturbation_type must have shape [N]")
    if not set(regimes) <= {"gentle", "large", "rotating"}:
        raise ValueError("Unknown pendulum regime")
    if split in {"train", "val"} and (np.any(regimes != "gentle") or np.any(perturbations != "none")):
        raise ValueError("train/val must contain only gentle, unperturbed trajectories")
    return SplitData(states, mask.astype(bool), ids[indices], regimes, perturbations,
                     times, split, metadata, portable_path(path))


class DynamicsModel(nn.Module):
    """Residual MLP: wrap-safe measured state -> next-state delta in physical units."""

    def __init__(self, hidden_sizes=(128, 128)):
        super().__init__()
        self.hidden_sizes = tuple(hidden_sizes)
        if not self.hidden_sizes or any(n <= 0 for n in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive widths")
        layers = []
        sizes = (3, *self.hidden_sizes)
        for before, after in zip(sizes, sizes[1:]):
            layers.extend([nn.Linear(before, after), nn.Tanh()])
        layers.append(nn.Linear(sizes[-1], 2))
        self.network = nn.Sequential(*layers)
        for name, value in {"input_mean": torch.zeros(3), "input_scale": torch.ones(3),
                            "delta_mean": torch.zeros(2), "delta_scale": torch.ones(2),
                            "state_scale": torch.ones(2)}.items():
            self.register_buffer(name, value)

    @staticmethod
    def encode(x):
        return torch.stack((x[..., 0].sin(), x[..., 0].cos(), x[..., 1]), dim=-1)

    def fit_normalization(self, x, delta):
        """Called once using valid adjacent transitions from train only."""
        encoded = self.encode(x)
        with torch.no_grad():
            self.input_mean.copy_(encoded.mean(0))
            self.input_scale.copy_(encoded.std(0, correction=0).clamp_min(1e-6))
            self.delta_mean.copy_(delta.mean(0))
            self.delta_scale.copy_(delta.std(0, correction=0).clamp_min(1e-6))
            self.state_scale.copy_(x.std(0, correction=0).clamp_min(1e-6))

    def forward(self, x):
        normalized = (self.encode(x) - self.input_mean) / self.input_scale
        return self.network(normalized) * self.delta_scale + self.delta_mean


class Ensemble:
    """Frozen CPU members; each rolls out independently from measured x [B,2]."""

    def __init__(self, models, dt: float):
        if not models or not np.isfinite(dt) or dt <= 0:
            raise ValueError("An ensemble needs members and a positive dt")
        self.models = list(models)
        self.dt = float(dt)
        self.checkpoint_sha256 = []
        for model in self.models:
            model.cpu().eval().requires_grad_(False)

    @torch.inference_mode()
    def forecast(self, x, k: int):
        """Return CPU tensors (mean, population std), both [B,k,2].

        No future observations or labels are accepted. Callers must supply a
        valid measured state (handle missing readings with the measurement mask).
        """
        x = torch.as_tensor(x, dtype=torch.float32, device="cpu")
        if x.ndim != 2 or x.shape[1] != 2 or not torch.isfinite(x).all():
            raise ValueError("x must contain finite measured states with shape [B,2]")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        trajectories = []
        for model in self.models:
            state, steps = x.clone(), []
            for _ in range(k):
                state = state + model(state)
                steps.append(state)
            trajectories.append(torch.stack(steps, dim=1))
        predictions = torch.stack(trajectories)
        if not torch.isfinite(predictions).all():
            raise ValueError("Non-finite model rollout")
        return predictions.mean(0), predictions.std(0, correction=0)

    def forecast_step(self, theta, omega, dt):
        """Adapter for existing referee windows; models have a fixed training dt."""
        if not np.isclose(dt, self.dt):
            raise ValueError(f"Forecast dt must match training dt={self.dt}")
        mean, _ = self.forecast([[theta, omega]], 1)
        return tuple(mean[0, 0].tolist())

    def forecast_last_step(self, theta_hist, omega_hist, dt):
        """Forecast the window's last sample from the latest valid reading before it.

        Dropped readings are NaN by contract, so roll out k steps from the last
        finite state instead of failing. With no valid earlier reading, return NaN
        (as StubForecaster does) and let the referee treat the window as missing.
        """
        if not np.isclose(dt, self.dt):
            raise ValueError(f"Forecast dt must match training dt={self.dt}")
        theta, omega = np.asarray(theta_hist, dtype=float), np.asarray(omega_hist, dtype=float)
        valid = np.flatnonzero(np.isfinite(theta[:-1]) & np.isfinite(omega[:-1]))
        if not valid.size:
            return (float("nan"), float("nan"))
        last = int(valid[-1])
        mean, _ = self.forecast([[theta[last], omega[last]]], len(theta) - 1 - last)
        return tuple(mean[0, -1].tolist())


@dataclass(frozen=True)
class TrainConfig:
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    hidden_sizes: tuple[int, ...] = (128, 128)
    epochs: int = 200
    batch_size: int = 1024
    learning_rate: float = 1e-3
    patience: int = 20
    min_delta: float = 1e-6
    threads: int = 1


def train_ensemble(train: SplitData, val: SplitData, config: TrainConfig | None = None):
    """Deterministic CPU training, independent seeds, val-only checkpoint choice."""
    config = config or TrainConfig()
    if train.split != "train" or val.split != "val":
        raise ValueError("Training requires train and val splits")
    if np.intersect1d(train.trajectory_id, val.trajectory_id).size:
        raise ValueError("train/val trajectory IDs overlap")
    if not np.isclose(train.dt, val.dt):
        raise ValueError("train/val dt mismatch")
    if (not config.seeds or len(set(config.seeds)) != len(config.seeds)
            or min(config.epochs, config.batch_size, config.patience, config.threads) < 1
            or not np.isfinite(config.learning_rate) or config.learning_rate <= 0
            or not np.isfinite(config.min_delta) or config.min_delta < 0):
        raise ValueError("Invalid training configuration")
    tx, ty = (torch.from_numpy(a) for a in train.transitions())
    vx, vy = (torch.from_numpy(a) for a in val.transitions())
    start = time.perf_counter()
    models, records = [], []
    prior_threads = torch.get_num_threads()
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(config.threads)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            for seed in config.seeds:
                torch.manual_seed(seed)
                model = DynamicsModel(config.hidden_sizes)
                model.fit_normalization(tx, ty)
                optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
                generator = torch.Generator().manual_seed(seed)
                best_loss, best_state, stale, best_epoch = float("inf"), None, 0, 0
                history = []
                for epoch in range(1, config.epochs + 1):
                    model.train()
                    order = torch.randperm(len(tx), generator=generator)
                    train_loss = 0.0
                    for batch in order.split(config.batch_size):
                        optimizer.zero_grad(set_to_none=True)
                        loss = (((model(tx[batch]) - ty[batch]) / model.delta_scale) ** 2).mean()
                        if not torch.isfinite(loss):
                            raise ValueError("Non-finite training loss")
                        loss.backward()
                        optimizer.step()
                        train_loss += float(loss.detach()) * len(batch)
                    model.eval()
                    with torch.no_grad():
                        val_loss = 0.0
                        for x, y in zip(vx.split(config.batch_size), vy.split(config.batch_size)):
                            error = (model(x) - y) / model.delta_scale
                            val_loss += float(error.square().mean()) * len(x)
                        val_loss /= len(vx)
                    if not np.isfinite(val_loss):
                        raise ValueError("Non-finite validation loss")
                    history.append({"epoch": epoch, "train_loss": train_loss / len(tx), "val_loss": val_loss})
                    if epoch == 1 or epoch % 10 == 0:
                        logger.info("Seed %s, epoch %s/%s: train loss %.6g, val loss %.6g",
                                    seed, epoch, config.epochs, train_loss / len(tx), val_loss)
                    if val_loss < best_loss - config.min_delta:
                        best_loss, best_epoch, stale = val_loss, epoch, 0
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    else:
                        stale += 1
                    if stale >= config.patience:
                        break
                model.load_state_dict(best_state)
                logger.info("Seed %s finished: best epoch %s, val loss %.6g (%s epochs run)",
                            seed, best_epoch, best_loss, epoch)
                models.append(model)
                records.append({"seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
                                "best_val_loss": best_loss, "history": history})
    finally:
        torch.set_num_threads(prior_threads)
        torch.use_deterministic_algorithms(prior_deterministic, warn_only=prior_warn_only)
    record = {"config": asdict(config), "members": records,
              "train_trajectory_ids": train.trajectory_id.tolist(),
              "val_trajectory_ids": val.trajectory_id.tolist(),
              "n_train_transitions": len(tx), "n_val_transitions": len(vx),
              "source": train.source, "data_metadata": train.metadata,
              "elapsed_seconds": time.perf_counter() - start,
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                              "numpy": np.__version__, "platform": platform.platform(),
                              "processor": platform.processor(), "device": "cpu"}}
    return Ensemble(models, train.dt), record


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_ensemble(directory, ensemble, training):
    """Save frozen weights + normalization and a hash manifest; refuse overwrite."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Checkpoint directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    for index, model in enumerate(ensemble.models):
        path = directory / f"member_{index}.pt"
        torch.save({"hidden_sizes": list(model.hidden_sizes), "state_dict": model.state_dict()}, path)
        checkpoints.append({"file": path.name, "sha256": sha256_file(path)})
    manifest = {"format_version": 1, "dt": ensemble.dt, "checkpoints": checkpoints,
                "normalization": {name: value.tolist() for name, value in ensemble.models[0].named_buffers()},
                "training": training}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    ensemble.checkpoint_sha256 = [c["sha256"] for c in checkpoints]


def verify_checkpoints(directory):
    """Read the manifest and verify hashes without constructing neural networks."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    for item in manifest["checkpoints"]:
        if Path(item["file"]).name != item["file"]:
            raise ValueError("Checkpoint file must be a local filename")
        path = directory / item["file"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"SHA256 mismatch: {path.name}")
    return manifest


def load_ensemble(directory):
    """Verify every SHA256 before CPU loading. Never resumes or changes weights."""
    directory = Path(directory)
    manifest = verify_checkpoints(directory)
    models = []
    for item in manifest["checkpoints"]:
        path = directory / item["file"]
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = DynamicsModel(saved["hidden_sizes"])
        model.load_state_dict(saved["state_dict"])
        models.append(model)
    ensemble = Ensemble(models, manifest["dt"])
    ensemble.checkpoint_sha256 = [c["sha256"] for c in manifest["checkpoints"]]
    return ensemble


def _sustained_crossing(errors, threshold, consecutive):
    for start in range(len(errors) - consecutive + 1):
        if np.all(errors[start:start + consecutive] > threshold):
            return start + 1  # forecast steps start at 1, not 0
    return None


def _bootstrap_ci(values, statistic, samples, rng):
    """Percentile CI resampling trajectory-level values, including tiny fixtures."""
    if len(values) == 0:
        return None
    if len(values) == 1:
        value = float(statistic(values))
        return [value, value]
    result = bootstrap((values,), statistic, n_resamples=samples,
                       method="percentile", vectorized=False, rng=rng)
    return [float(result.confidence_interval.low), float(result.confidence_interval.high)]


def model_report(ensemble, data: SplitData, threshold=0.25, sustained_steps=3,
                 bootstrap_samples=1000, seed=0):
    """Held-out clean-regime report from val/cal; no model selection on cal.

    Error is L2 of [wrapped angle error, speed error] divided by train state
    std. Reliability horizon is the FIRST step of three successive errors
    above 0.25. Uncrossed horizons are right-censored at the available horizon;
    summaries use these caps and explicitly report censoring. Bootstrap draws
    whole trajectories. Clean runs isolate regime generalization from faults.
    """
    if data.split not in {"val", "cal"}:
        raise ValueError("Model reports require held-out val or cal; test is locked")
    if not np.isclose(data.dt, ensemble.dt):
        raise ValueError("Report dt differs from training dt")
    if threshold <= 0 or not np.isfinite(threshold) or sustained_steps < 1 or bootstrap_samples < 1:
        raise ValueError("Invalid report settings")
    scale = ensemble.models[0].state_scale.numpy()
    regimes = {}
    rng = np.random.default_rng(seed)
    for regime in ("gentle", "large", "rotating"):
        rows = np.flatnonzero((data.regime == regime) & (data.perturbation_type == "none"))
        details = []
        for i in rows:
            states = data.states[i]
            valid = data.mask[i].all(-1) & np.isfinite(states).all(-1)
            pairs = valid[:-1] & valid[1:]
            if not pairs.any():
                continue
            mean, _ = ensemble.forecast(states[:-1][pairs], 1)
            error = mean[:, 0].numpy() - states[1:][pairs]
            error[:, 0] = angle_difference(mean[:, 0, 0].numpy(), states[1:, 0][pairs])
            squared = np.sum((error / scale) ** 2, axis=-1)
            horizon, crossing = 0, None
            if valid[0]:
                missing = np.flatnonzero(~valid)
                horizon = int(missing[0] - 1) if missing.size else len(states) - 1
                if horizon > 0:
                    rollout, _ = ensemble.forecast(states[None, 0], horizon)
                    diff = rollout[0].numpy() - states[1:horizon + 1]
                    diff[:, 0] = angle_difference(rollout[0, :, 0].numpy(), states[1:horizon + 1, 0])
                    crossing = _sustained_crossing(np.linalg.norm(diff / scale, axis=-1), threshold, sustained_steps)
            details.append({"trajectory_id": int(data.trajectory_id[i]),
                            "one_step_normalized_mse": float(squared.mean()),
                            "angle_mse": float(np.mean(error[:, 0] ** 2)),
                            "omega_mse": float(np.mean(error[:, 1] ** 2)),
                            "n_transitions": int(pairs.sum()),
                            "first_sustained_crossing_step": crossing,
                            "observed_horizon_steps": horizon,
                            "reliability_horizon_steps": crossing if crossing is not None else horizon,
                            "right_censored": crossing is None})
        if not details:
            regimes[regime] = {"status": "unavailable", "n_trajectories": 0,
                               "reason": "No clean trajectories with valid adjacent measurements"}
            continue
        mse = np.array([d["one_step_normalized_mse"] for d in details])
        horizons = np.array([d["reliability_horizon_steps"] for d in details if d["observed_horizon_steps"] > 0])
        rmse_ci = _bootstrap_ci(mse, lambda x: np.sqrt(x.mean()), bootstrap_samples, rng)
        horizon_ci = _bootstrap_ci(horizons, np.median, bootstrap_samples, rng)
        regimes[regime] = {
            "status": "measured", "n_trajectories": len(details),
            "n_transitions": sum(d["n_transitions"] for d in details),
            "one_step_normalized_rmse": float(np.sqrt(mse.mean())),
            "one_step_normalized_rmse_ci95": rmse_ci,
            "one_step_angle_rmse_rad": float(np.sqrt(np.mean([d["angle_mse"] for d in details]))),
            "one_step_omega_rmse_rad_s": float(np.sqrt(np.mean([d["omega_mse"] for d in details]))),
            "reliability_horizon_steps_median": float(np.median(horizons)) if horizons.size else None,
            "reliability_horizon_seconds_median": float(np.median(horizons) * data.dt) if horizons.size else None,
            "reliability_horizon_steps_ci95": horizon_ci,
            "n_horizon_trajectories": int(horizons.size),
            "n_right_censored": sum(d["right_censored"] for d in details if d["observed_horizon_steps"] > 0),
            "trajectories": details,
        }
    return {"split": data.split, "source": data.source, "data_metadata": data.metadata,
            "dt": data.dt, "threshold": threshold, "sustained_steps": sustained_steps,
            "error_definition": "L2([wrapped angle error, omega error] / train state std)",
            "state_scale": scale.tolist(), "selection": "perturbation_type == none, per regime",
            "horizon_definition": "First step of sustained crossing; uncrossed horizons capped at last contiguous observation",
            "aggregation": "Equal trajectory weight; capped horizon medians are lower bounds when censored",
            "bootstrap": {"unit": "trajectory", "samples": bootstrap_samples, "seed": seed, "confidence": 0.95},
            "checkpoint_sha256": ensemble.checkpoint_sha256, "regimes": regimes}


class StubForecaster:
    def __init__(self, g: float = 9.81, L: float = 1.0, b: float = 0.0, m: float = 1.0):
        self.g, self.L, self.b, self.m = g, L, b, m

    def forecast_step(self, theta: float, omega: float, dt: float) -> tuple[float, float]:
        domega = -(self.g / self.L) * theta - (self.b / (self.m * self.L**2)) * omega
        next_omega = omega + domega * dt
        next_theta = theta + omega * dt
        return next_theta, next_omega

    def forecast_last_step(self, theta_hist, omega_hist, dt: float) -> tuple[float, float]:
        """Predict the final point in the window from the one before it."""
        return self.forecast_step(theta_hist[-2], omega_hist[-2], dt)
