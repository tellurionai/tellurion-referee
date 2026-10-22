"""CLI: python -m tellurion_sim {generate, scenario, check} (plan-reality-engine.md §6)."""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import yaml

from . import integrate
from .check import check_dataset
from .dataset import (
    RunSpec,
    generate,
    load_config,
    metadata,
    save_dataset,
    simulate_run,
)
from .export import write_run_csv
from .perturbations import PERTURBATION_TYPES, SEVERITIES, expected_verdict
from .systems.pendulum import REGIMES

DEFAULT_CONFIG = "configs/pendulum.yaml"


def cmd_generate(a) -> int:
    cfg = load_config(a.config)
    if cfg["system"] != a.system:
        raise SystemExit(f"config is for {cfg['system']!r}, not {a.system!r}")
    t0 = time.time()
    ds = generate(cfg, a.seed, workers=a.workers)
    paths = save_dataset(ds, metadata(cfg, a.seed), a.out)
    print(f"generated {len(ds['trajectory_id'])} trajectories x {len(ds['times'])} samples in {time.time() - t0:.1f} s")
    print("\n".join(f"wrote {p}" for p in paths))
    return 0


def cmd_scenario(a) -> int:
    cfg = load_config(a.config)
    if a.perturb == "none":
        subtype, severity, onset = "none", "none", -1
    else:
        if a.subtype is None:
            raise SystemExit("--subtype is required unless --perturb none")
        subtype, severity, onset = a.subtype, a.severity, a.onset
    spec = RunSpec(0, "scenario", a.regime, a.perturb, subtype, severity, onset)
    run = simulate_run(cfg, spec, np.random.default_rng(a.seed))
    from .systems import get_system
    labels = {"perturbation_type": a.perturb, "perturbation_subtype": subtype, "severity": severity,
              "onset_step": onset, "expected_verdict": expected_verdict(a.regime, a.perturb)}
    out = write_run_csv(a.csv, get_system(a.system), integrate.time_grid(cfg["dt"], cfg["n_steps"]), run, labels)
    print(f"wrote {out} ({labels['expected_verdict']})")
    return 0


def cmd_check(a) -> int:
    ok, lines = check_dataset(a.path, include_test=a.include_test)
    print("\n".join(lines))
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tellurion_sim")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="full labelled dataset -> <out>.npz + <out>.metadata.json")
    g.add_argument("--system", default="pendulum")
    g.add_argument("--config", default=DEFAULT_CONFIG)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", required=True, help="output prefix, e.g. data/pendulum_v1")
    g.add_argument("--workers", type=int, default=None, help="processes (default: all CPUs)")
    g.set_defaults(fn=cmd_generate)

    s = sub.add_parser("scenario", help="one labelled run -> CSV")
    s.add_argument("--system", default="pendulum")
    s.add_argument("--config", default=DEFAULT_CONFIG)
    s.add_argument("--regime", choices=REGIMES, default="gentle")
    s.add_argument("--perturb", choices=PERTURBATION_TYPES, default="none")
    s.add_argument("--subtype")
    s.add_argument("--severity", choices=("none",) + SEVERITIES, default="med")
    s.add_argument("--onset", type=int, default=80)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--csv", required=True)
    s.set_defaults(fn=cmd_scenario)

    c = sub.add_parser("check", help="sanity report; non-zero exit on any failure")
    c.add_argument("path", help="dataset prefix or .npz")
    c.add_argument("--include-test", action="store_true", help="also read the locked test file")
    c.add_argument("--seed", type=int, default=0, help="unused; accepted for a uniform CLI")
    c.set_defaults(fn=cmd_check)

    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except (ValueError, KeyError, OSError, yaml.YAMLError) as e:  # bad input or files; bugs keep their traceback
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
