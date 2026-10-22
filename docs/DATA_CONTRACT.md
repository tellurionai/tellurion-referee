# Reality engine data contract

This is the shared interface for the reality engine, model, physics fit,
referee, evaluation, and replay exporter. It documents the implementation
merged in PR #65 and addresses [issue #6](https://github.com/tellurionai/tellurion-referee/issues/6).
The executable schema and validation live in `tellurion_sim/contract.py`;
labels come from `tellurion_sim.perturbations.expected_verdict`.

Both **angle theta and angular velocity omega are independently measured**.
Sensor faults corrupt **theta only**. Omega remains equal to true omega and
valid, including during angle dropout. This is the implemented sensor model;
Core team acknowledgment is still required and is not implied by this document.

## Files and dimensions

For an output prefix `data/pendulum_v1`, the existing writer produces:

| File | Contents |
| --- | --- |
| `pendulum_v1.npz` | Development trajectories: `train`, `val`, `cal` |
| `pendulum_v1.test.npz` | Locked `test` trajectories, stored separately |
| `pendulum_v1.metadata.json` | Shared generation settings and provenance |

The separate test archive is an existing implementation detail added to
protect the locked split; do not combine it into the development archive.
Each archive requires the same 15 core arrays. Readers ignore additional fields
without loading them, so an additive field does not break existing consumers.
The writer preserves additional arrays with a leading trajectory dimension
`[N,...]` and partitions them with the core fields. Extension arrays must use
non-object NumPy dtypes; scalar settings belong in the metadata sidecar.
`N` is the number of trajectories in
that archive, `T` the number of integration steps, `S` the state dimension,
and `P` the parameter dimension. A split archive may contain zero trajectories.
All trajectories share one time grid containing `T + 1` samples, including
the initial state at step zero. Strings are NumPy Unicode arrays (`dtype.kind
== "U"`), with no fixed required width. Object arrays and pickle are forbidden.

| Array | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `times` | `[T+1]` | `float64` | Uniform sample times in seconds, starting at zero |
| `true_states` | `[N,T+1,S]` | `float32` | Simulated ground truth |
| `measured_states` | `[N,T+1,S]` | `float32` | Sensor readings, with NaN only for dropped readings |
| `measured_mask` | `[N,T+1,S]` | `uint8` | Exactly 1 for valid readings, 0 for missing readings |
| `true_energy` | `[N,T+1]` | `float32` | `system.energy` using parameters active at that sample |
| `nominal_params` | `[N,P]` | `float32` | Parameters supplied to the model |
| `actual_params` | `[N,P]` | `float32` | Parameters used from onset onward; see timing below |
| `trajectory_id` | `[N]` | `int64` | Unique nonnegative ID within one generated dataset |
| `split` | `[N]` | Unicode | `train`, `val`, `cal`, `test` |
| `regime` | `[N]` | Unicode | `gentle`, `large`, `rotating` |
| `perturbation_type` | `[N]` | Unicode | `none`, `param_shift`, `sensor_fault`, `unseen_force` |
| `perturbation_subtype` | `[N]` | Unicode | Subtype listed below; `none` for clean runs |
| `severity` | `[N]` | Unicode | `none`, `low`, `med`, `high` |
| `onset_step` | `[N]` | `int32` | -1 for `none`; otherwise an index in `[0,T]` |
| `expected_verdict` | `[N]` | Unicode | `trust`, `model_error`, `instrument_fault`, `discovery` |

## Pendulum conventions

State order is `[theta, omega]`, units `[rad, rad/s]` (`S=2`). Parameter order
is `[g, L, m, b]`, units `[m/s^2, m, kg, N*m*s/rad]` (`P=4`). Never infer
parameter order from alphabetical sorting. Nominal values are
`[9.81, 1.0, 1.0, 0.0]`. All true states, energies, and parameters are finite;
`g`, `L`, and `m` are positive, and damping `b` is nonnegative.

Theta is zero hanging down and positive counterclockwise. Angles are unwrapped
radians, including rotations. With the pivot at `(0,0)`, bob position is
`x = L*sin(theta)`, `y = -L*cos(theta)`. Energy is
`0.5*m*L**2*omega**2 + m*g*L*(1-cos(theta))` in joules.

Initial regimes: `gentle` has `abs(theta0) <= 0.6` and `abs(omega0) <= 0.8`;
`large` has `abs(theta0)` in `[1.5,2.8]` and `abs(omega0) <= 0.5`;
`rotating` has energy above the nominal over-the-top threshold `2*m*g*L`.

## Perturbations, onset, and labels

| Type | Allowed subtypes | Expected verdict |
| --- | --- | --- |
| `none` | `none` | `trust` in `gentle`; `model_error` in `large` or `rotating` |
| `param_shift` | `length`, `gravity`, `damping` | `model_error` |
| `sensor_fault` | `bias`, `drift`, `noise`, `scale`, `stuck`, `dropout` | `instrument_fault` |
| `unseen_force` | `constant_torque`, `magnet`, `periodic_drive`, `impulse` | `discovery` |

These are trajectory labels, not per-step verdicts. A perturbation's label
describes the run even before onset. Evaluation must account for onset when
calculating detection delays and pre-onset false alarms. Clean `large` and
`rotating` runs have no onset but are out of the model's training regime.

For `none`, subtype and severity are `none` and onset is -1. Generated
perturbations use `low`, `med`, or `high`; the single-scenario API also permits
severity `none` as zero strength. A zero-strength scenario retains its requested
perturbation label; it does not change the label rule above.

At onset index `k`, parameter shifts take effect and sensor faults begin.
Samples strictly before `k` match the clean run with the same initial state.
The state at `k` is continuous for parameter shifts and continuous forces;
the following interval uses the changed dynamics. An impulse changes omega
at sample `k` itself, by `J/(m*L**2)`. An onset of zero means active from the
initial sample. The integrator restarts at every output sample.

`actual_params[i]` stores the post-onset vector, **not** a time series. For
`param_shift`, use `nominal_params[i]` before onset and `actual_params[i]` at
and after onset when reconstructing physics, positions, or energy. For other
types these two vectors are identical. External forces are not additional
entries in the parameter vector.

Sensor faults never change true motion or true energy. Outside `sensor_fault`,
measurements equal truth and all mask entries are 1. Before sensor onset,
measurements also equal truth. Only angle `dropout` creates missing samples:
`measured_states[i,k,0] = NaN` and `measured_mask[i,k,0] = 0`. Valid readings
are finite. Do not turn missing angles into zero or drop the healthy omega
channel. CSV uses an empty angle cell for a missing reading.

## Splits and leakage

Splitting is by whole `trajectory_id`, never by timestep. IDs are unique
across the development and locked archives from one generation. The same
seed **and full configuration** reproduce the same arrays and IDs across
worker counts. IDs are not globally unique across datasets; changing split
sizes or configuration may change the identity associated with an ID.

- `train` and `val` contain only clean `gentle` trajectories.
- `cal` is for threshold fitting and development evaluation.
- `test` is locked for the single final evaluation through
  `scripts/evaluate.py --split test` when that pipeline is available.
- Do not tune models, thresholds, strengths, or scenarios against locked data.

Defaults are 2,000 train / 400 val / 600 cal / 1,000 test, `dt=0.05` seconds,
and `T=200` (10 seconds). For cal/test the planner reserves 20% additional
clean gentle runs; the rest cycle through the 30 regime/type/severity cells
(3 regimes times 1 clean plus 9 perturbed cells). It then shuffles with the
seeded RNG. Counts differ by at most one in that cycle; subtypes are sampled
uniformly, not guaranteed balanced. Small configs need not cover every cell.
Default onset is sampled uniformly from integer steps 40 through 120 inclusive.

At decision time, construct an explicit referee history from measurements,
masks, nominal parameters, timing, and model forecasts. Do not pass the dataset
dictionary wholesale. Ground truth, true energy, actual parameters, regimes,
split labels, perturbation labels, severity, onset, and expected verdict are
offline generation/evaluation information, not decision inputs.

## Metadata

The writer records the following keys in the shared JSON sidecar:

| Keys | Contents |
| --- | --- |
| `system` | `pendulum` |
| `state_names`, `state_units`, `param_names`, `param_units` | Ordered channel/parameter descriptions |
| `nominal_params` | Nominal parameter mapping |
| `dt`, `n_steps`, `horizon_s` | Sample interval, integration-step count, duration (`dt*n_steps`) |
| `solver` | `method=DOP853`, `rtol=1e-9`, `atol=1e-11`, restart policy |
| `seed`, `git_commit`, `generated_at` | RNG seed, generating commit (or `unknown`), UTC ISO timestamp |
| `severity_table` | Exact numeric strengths for every type/subtype/severity from config |
| `magnet_phi_m`, `periodic_drive_w` | Magnet direction and drive frequency |
| `onset` | Configured inclusive onset range |
| `split_sizes`, `clean_gentle_frac`, `split_rules` | Generation counts and selection policy |
| `sensor_model`, `verdict_rule` | Human-readable assumptions and labeling rule |
| `files` | `dev` and `test` archive basenames, added by the writer |

Strengths and units are defined in `configs/pendulum.yaml`. In particular,
length and gravity strengths are multipliers, damping is an absolute value,
drift is the final accumulated angle offset, and scale is the fractional gain
error (`theta*(1+s)`). Stuck uses a boolean strength; periodic drive uses elapsed
time since onset. Preserve this metadata with the archives when handing off.
Generation timestamps and the producing commit are provenance, not part of
the deterministic-array guarantee.

## Using the existing API

```python
from tellurion_sim.dataset import load_dataset

train = load_dataset("data/pendulum_v1", splits=("train",))
cal = load_dataset("data/pendulum_v1.npz", splits=("cal",))
# Default splits are train/val/cal; the locked archive is never opened.
```

`save_dataset(ds, meta, prefix)` validates before writing and returns the
development, locked, and metadata paths. `load_dataset` returns the core NumPy
arrays, requires the metadata sidecar, and validates each archive once before
filtering. Metadata must identify the system, ordered state and parameter names,
sample interval, and step count. Missing or invalid metadata is an error.
Combining archives additionally checks time-grid agreement and ID uniqueness.
A misplaced locked row in a development archive is an
error, not a row to silently filter away. Explicitly requesting `test` is
supported by the loader for final evaluation; this API is not an authorization
boundary. The legacy `check --include-test` option also reads it and must not
be used during development.

For in-memory data, `dataset_errors(ds, meta)` returns contract diagnostics and
`validate_dataset(ds, meta)` raises `ValueError` on violations. Both live in
`tellurion_sim.contract`. Shared checks cover the required arrays; system-specific
rules come from `System.dataset_errors`, with pendulum parameter bounds and
healthy-omega assumptions defined by `Pendulum`. Additional systems can define
their own parameter and sensor rules without inheriting pendulum column indices.
The checker reports the same contract diagnostics individually and additionally
measures pendulum energy and period invariants. On invalid archives it retains
the seed/commit header and readable label counts; missing metadata is reported
as a failure with provenance unavailable.
The older `schema_errors` import from `tellurion_sim.check` remains available.
Legacy stub data is a development fixture, not the production handoff: it has
different metadata and stores synthetic `test` rows in its single archive.

```powershell
python -m pip install -r requirements.txt
python -m tellurion_sim generate --system pendulum --config configs/pendulum.yaml --seed 0 --out data/pendulum_v1
python -m tellurion_sim check data/pendulum_v1
python -m tellurion_sim scenario --system pendulum --regime gentle --perturb sensor_fault --subtype bias --severity med --onset 80 --seed 3 --csv out/pendulum_bias_med.csv
python -m pytest -q
```

## Single-run CSV

`tellurion_sim.export.write_run_csv` writes one row per sample, with this exact
column order (the line below is one CSV header):

```csv
time_s,true_angle_rad,measured_angle_rad,measured_valid,predicted_angle_rad,confidence_0to1,true_x,true_y,pred_x,pred_y,perturbation_type,perturbation_subtype,severity,onset_step,expected_verdict
```

`measured_valid` is the angle-channel mask. Predictions, confidence, and predicted
positions are empty for the model/export pipeline to fill. Labels repeat on
each row; `onset_step` is an index, not seconds. True positions use the length
active at each sample. This CSV is simulator output, not the demo replay JSON.

## Coordination

Changes to names, shapes, units, labels, sensor assumptions, or split policy
must be described in the PR and communicated to all consuming teams before
adoption. Core acknowledgment (model, referee, physics fit, evaluation) remains
pending; no acknowledgment or historical deadline completion is claimed here.
