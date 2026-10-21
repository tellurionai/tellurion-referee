"""Train/freeze the five-seed world model, or report on existing frozen weights.

Examples (run from the repository root):
    python scripts/train.py --data data/pendulum_v1.npz
    python scripts/train.py --data data/pendulum_v1.npz --report-only
    python scripts/train.py --config configs/base.yaml

Optional YAML schema (paths relative to the working directory; CLI overrides):
    data: data/pendulum_v1.npz
    out: artifacts/world_model
    report: results/model_report.json
    report_split: cal
    bootstrap_samples: 1000
    model:
      seeds: [0, 1, 2, 3, 4]
      hidden_sizes: [128, 128]
      epochs: 200
      batch_size: 1024
      learning_rate: 0.001
      patience: 20
      min_delta: 0.000001
      threads: 1

Only clean gentle train/val are used for fitting. Reports use cal when available,
otherwise val (missing regimes remain unavailable). --report-only lets the full
cal handoff arrive later without retraining or changing the frozen checkpoints.
The test split is never selected. Existing checkpoint directories are protected.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from referee.model import (  # noqa: E402
    TrainConfig, load_split, train_ensemble, save_ensemble, load_ensemble,
    model_report, sha256_file,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--report-split", choices=("val", "cal"))
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--hidden-sizes", nargs="+", type=int)
    parser.add_argument("--bootstrap-samples", type=int)
    args = parser.parse_args(argv)
    settings = {}
    if args.config:
        import yaml
        settings = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        if not isinstance(settings, dict):
            parser.error("Config must be a YAML mapping")
    options = asdict(TrainConfig())
    model_settings = settings.get("model", {})
    if not isinstance(model_settings, dict) or set(model_settings) - set(options):
        parser.error("Unknown model configuration fields; see --help for the schema")
    options.update(model_settings)
    for name in options:
        override = getattr(args, name, None)
        if override is not None:
            options[name] = override
    options["seeds"] = tuple(options["seeds"])
    options["hidden_sizes"] = tuple(options["hidden_sizes"])
    config = TrainConfig(**options)
    data_path = Path(args.data or settings.get("data", "data/pendulum_v1.npz"))
    if data_path.suffix != ".npz":
        data_path = data_path.with_suffix(".npz")
    out = Path(args.out or settings.get("out", "artifacts/world_model"))
    report_path = Path(args.report or settings.get("report", "results/model_report.json"))
    report_split = args.report_split or settings.get("report_split")
    bootstrap_samples = args.bootstrap_samples if args.bootstrap_samples is not None else settings.get("bootstrap_samples", 1000)
    if report_split is not None and report_split not in {"val", "cal"}:
        parser.error("Reports allow only val or cal; test is locked")
    if bootstrap_samples < 1:
        parser.error("bootstrap_samples must be positive")
    try:
        if report_path.resolve().is_relative_to(out.resolve()):
            raise ValueError("Write reports outside the frozen checkpoint directory")
        # Inspect split labels only. No test states or truth arrays are loaded.
        with np.load(data_path, allow_pickle=False) as arrays:
            available = set(arrays["split"].astype(str))
        if report_split is None:
            report_split = "cal" if "cal" in available else "val"
        if not args.report_only and out.exists() and any(out.iterdir()):
            raise FileExistsError(f"Checkpoint directory is not empty: {out}. Use --report-only or a new --out.")
        if args.report_only:
            ensemble = load_ensemble(out)
        else:
            train, val = load_split(data_path, "train"), load_split(data_path, "val")
            print(f"Training {len(config.seeds)} members: {len(train.trajectory_id)} train / "
                  f"{len(val.trajectory_id)} val trajectories on CPU", flush=True)
            ensemble, training = train_ensemble(train, val, config)
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                    cwd=Path(__file__).resolve().parents[1], check=False)
            training["code_commit"] = commit.stdout.strip() if commit.returncode == 0 else None
            status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=False,
                                    cwd=Path(__file__).resolve().parents[1])
            training["code_dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
            training["data_sha256"] = sha256_file(data_path)
            save_ensemble(out, ensemble, training)
            print(f"Saved frozen checkpoints to {out}", flush=True)
        report = model_report(ensemble, load_split(data_path, report_split), bootstrap_samples=bootstrap_samples)
        report["manifest"] = str((out / "manifest.json").resolve())
        # Recheck saved weights after reporting, including in report-only mode.
        load_ensemble(out)
        report["checkpoint_hashes_verified"] = True
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Wrote {report_split} model report to {report_path}")
        for regime, measured in report["regimes"].items():
            if measured["status"] == "unavailable":
                print(f"  {regime}: unavailable")
            else:
                print(f"  {regime}: N={measured['n_trajectories']}, "
                      f"normalized one-step RMSE={measured['one_step_normalized_rmse']:.6g}, "
                      f"median capped horizon={measured['reliability_horizon_steps_median']} steps")
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
