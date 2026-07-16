# Sim2Real Bridge

This document explains, concretely and without overclaiming, what "Sim2Real"
means in this repo: there is no physical packing rig, so the "real world" is
represented by a second, wider domain-randomization profile run through the
same PyBullet simulator. This is a legitimate technique for *estimating
sim-overfitting risk*, but it is not real-world validation, and this doc is
explicit about that distinction throughout.

## Two randomization profiles

`envs/domain_randomization.py` defines the seam:

```python
VALID_MODES = ("train", "real_proxy")

def get_randomization_profile(env_config: dict, mode: str) -> dict:
    if mode not in VALID_MODES:
        raise ValueError(...)
    return env_config["domain_randomization"][mode]
```

Both profiles live under `domain_randomization:` in `configs/env.yaml`, and
each is a dict of ranges that `pybullet_sim.py::DropSimulator.simulate_drop`
samples from (via `rng.uniform`/the fixed `pose_noise_m`) on **every** drop
check — for the box being placed and, for `pose_noise_m`, for the already
committed boxes' positions too:

```yaml
domain_randomization:
  train:
    friction_range: [0.4, 0.8]
    mass_scale_range: [0.9, 1.1]
    restitution_range: [0.0, 0.1]
    pose_noise_m: 0.0
  real_proxy:                      # wider range standing in for unmodeled real-world variation
    friction_range: [0.2, 1.0]
    mass_scale_range: [0.7, 1.4]
    restitution_range: [0.0, 0.3]
    pose_noise_m: 0.01
```

| Parameter | `train` | `real_proxy` | What it perturbs |
|---|---|---|---|
| `friction_range` | `[0.4, 0.8]` | `[0.2, 1.0]` | `lateralFriction` on every spawned box (`_spawn_box`) |
| `mass_scale_range` | `[0.9, 1.1]` | `[0.7, 1.4]` | multiplier on the density-derived mass (`_DENSITY_KG_PER_M3 = 300.0 kg/m^3`) of every box |
| `restitution_range` | `[0.0, 0.1]` | `[0.0, 0.3]` | `restitution` (bounciness) on every spawned box |
| `pose_noise_m` | `0.0` (none) | `0.01` | Gaussian noise (`rng.normal(0, pose_noise, size=3)`) added to each *already-placed* box's position before it's re-spawned for this drop check |

`train` is the distribution both PPO agents actually train under —
`training/train.py::train()` hardcodes
`PackingEnv(env_config, randomization_mode="train", ...)`. `real_proxy` is
**never used during training**; it is exclusively an evaluation-time
distribution, deliberately widened to simulate the kind of physical
variation a training run inside a single narrow-range simulator would never
see: looser/tighter friction, real cartons' mass varying with contents,
non-zero settling noise on stacks that were placed on a previous, noisy
drop.

**Be precise about what this is not**: `real_proxy` is not measured from any
physical rig, sensor log, or warehouse trial. It's a synthetic stress-test —
a proxy in the literal sense — chosen to be *wider* than the training
distribution so that a policy which has quietly overfit to the exact
`train` physics parameters will visibly perform worse under it. A real
Sim2Real pipeline would replace `real_proxy` with actual robot/rig
telemetry; this repo has none, so it substitutes a harder synthetic
distribution instead. See "Swapping in real measurements" below for what
that replacement would look like.

## Running the same model under both: `benchmark.py`

`evaluation/benchmark.py::run_benchmark(packer_model, physics_model,
env_config, randomization_mode, num_episodes, seed_base, reward_shaping)`
drives a fixed number of full episodes through `PackingEnv` — constructed
with the given `randomization_mode` — using both models' `.predict(...,
deterministic=True)` outputs (no exploration noise at eval time), and
aggregates:

- `box_utilization_pct` (mean `episode_summary["utilization_pct"]` across
  episodes) via `evaluation/metrics.py`.
- `stability_success_rate_pct` from the sequence of `outcome` strings
  emitted in `infos["physics"]["outcome"]` on every physics turn (see
  `docs/MARL_DESIGN.md` for the 5 outcome names). Note its docstring:
  it's the fraction of *accepted* placements that were actually stable,
  not a fraction over all items — rejected items don't count against it,
  since a reject only costs utilization, tracked separately.

`mlops/challenger.py::_evaluate_version` calls `run_benchmark` **twice** for
a given model version — once with `"train"`, once with `"real_proxy"` — on
identical seeds (`seed_base` fixed via `env.reset(seed=seed_base + i)` for
`i` in `range(num_episodes)`), so any utilization difference between the two
runs is attributable to the randomization profile, not to different item
sequences.

## Computing the gap: `sim2real_gap_pct`

```python
def sim2real_gap_pct(sim_utilization_pct: float, real_proxy_utilization_pct: float) -> float:
    """Positive gap means the policy looks better in sim than under the real-world-proxy
    randomization range -- the classic Sim2Real overfitting signature."""
    return sim_utilization_pct - real_proxy_utilization_pct
```

A positive gap means the packer/physics pair achieves higher utilization
under `train`-distribution physics than under the wider `real_proxy`
distribution — a signature that the policy has learned to lean on physics
regularities (a specific friction band, a specific mass scale) that won't
hold up outside the narrow training range. This gap is one of the three
numbers reported in the `EvalReport` dataclass (`mlops/challenger.py`) and
is one of the three Champion/Challenger promotion-gate conditions (see
`docs/MLOPS.md`).

## `real_world_reference.csv`: what it actually is

`evaluation/real_world_reference.csv` exists to name where genuine
real-world utilization targets would live once measured. Its header is
explicit that it is not that yet:

```
# ILLUSTRATIVE PLACEHOLDER DATA -- not measurements from a real warehouse or rig.
#
# This repo has no physical packing cell to benchmark against (see docs/SIM2REAL.md), so
# this file stands in for wherever your real box-utilization measurements would go. The
# numbers below are round, plausible targets loosely informed by published bin-packing /
# palletizing utilization ranges (typically ~65-85% for mixed-carton loads), NOT audited
# real-world data. Replace this file's contents with your own measurements to make the
# Sim2Real benchmark in evaluation/benchmark.py meaningful.
```

The three rows (`mixed_small_boxes` → 72.0%, `uniform_medium_boxes` →
80.0%, `irregular_load` → 65.0%) are round numbers picked to be plausible
against published palletizing literature — they are not derived from any
run of this codebase, any sensor, or any audited process. Nothing in
`evaluation/`, `mlops/`, or `training/` currently reads this CSV
programmatically — it is reference/documentation data for a human to
compare a `benchmark.py` result against, not a wired-up gate input.

### Swapping in genuine measurements

To make this file (and by extension, any comparison against it) meaningful:

1. Instrument a real packing process — physical or from warehouse logs — and
   record, per scenario, the achieved box-utilization percentage using the
   same definition as `box_utilization_pct` (utilized volume / bin volume).
2. Replace the `target_box_utilization_pct` values in
   `real_world_reference.csv` with those measured numbers, and update or
   remove the header disclaimer once the data is genuinely audited.
3. Optionally, tighten `real_proxy`'s domain-randomization ranges in
   `configs/env.yaml` to match measured physical variation (e.g. real
   friction-coefficient spread for your cartons/conveyor material, real
   mass variance for your SKU mix) instead of the current placeholder
   ranges, so `sim2real_gap_pct` reflects your actual hardware rather than a
   generic "wider band" heuristic.
4. If a genuine automated real-world comparison is wanted, wire the CSV (or
   its replacement) into `mlops/challenger.py`'s gate as a fourth condition,
   rather than leaving it as a human-facing reference file.

Until those steps happen, treat every "Sim2Real gap" number this repo
produces as a *simulator-only robustness proxy*, not a validated real-world
performance guarantee.

## Calibrating `real_proxy` from real data: `evaluation/calibration.py`

Step 3 above ("tighten the ranges to match measured physical variation") has
a concrete mechanism rather than being purely manual: `calibrate()` fits the
`DropSimulator` physics point (friction, mass_scale, restitution) that best
reproduces a set of observed real placement outcomes, using the
Cross-Entropy Method (CEM) — a derivative-free search well suited to this
objective (agreement-with-logs is discrete and noisy, so there's no useful
gradient through the physics engine to follow).

Collect `PlacementLog`s from a real cell (or an operator's scripted trial
run) — each one records the stack it was dropped onto, the candidate box,
and whether it was actually stable:

```python
from marl_packing.evaluation.calibration import PlacementLog, calibrate
from marl_packing.envs.pybullet_sim import PlacedBox

logs = [
    PlacementLog(existing_boxes=[...], candidate=PlacedBox(position=..., dims=...), stable=True),
    # ...one per real placement attempt...
]
result = calibrate(logs, env_config, iterations=8, population=24)
print(result.params, result.agreement)  # CalibratedParams(friction=..., mass_scale=..., restitution=...)
```

`result.params.as_randomization_profile()` returns a degenerate (zero-width)
domain-randomization profile pinned to the calibrated point — usable
directly as a data-grounded `real_proxy` in place of the current hand-picked
range, or as a center to build a calibrated range around. `result.history`
tracks the best-agreement-so-far per CEM iteration, useful for checking the
search actually converged before trusting the result.
