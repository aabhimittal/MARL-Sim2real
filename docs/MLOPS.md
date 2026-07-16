# MLOps Loop: Tracking, Registry, and Champion/Challenger Promotion

This document walks the full path a trained model takes: from MLflow
experiment logging during `training/train.py`, through registration in the
local JSON model registry (`mlops/registry.py`), through the
Champion/Challenger promotion gate (`mlops/challenger.py`), to the
`mlops/deploy.py` CLI that ties evaluation and promotion together.

## 1. MLflow experiment tracking

`training/train.py::train()` wraps the entire alternating-round training
loop in one MLflow run:

```python
mlflow.set_tracking_uri(mlflow_tracking_uri)   # default "./mlruns"
mlflow.set_experiment(experiment_name)          # default "marl-packing"
with mlflow.start_run() as run:
    mlflow.log_params({
        "full_scale": full,
        "n_rounds": n_rounds,
        "timesteps_per_round": timesteps_per_round,
        **{f"packer_ppo_{k}": v for k, v in packer_config["ppo"].items()},
        **{f"physics_ppo_{k}": v for k, v in physics_config["ppo"].items()},
    })
    ...
```

So every run logs, as MLflow **params**: whether it was a full-scale or
smoke run, the round count and timesteps-per-round from
`configs/train_packer.yaml`'s `schedule` block, and every PPO hyperparameter
for both agents (`configs/train_packer.yaml`'s and
`configs/train_physics.yaml`'s `ppo:` blocks — `learning_rate`, `n_steps`,
`batch_size`, `n_epochs`, `gamma`, `gae_lambda`, `clip_range`, `ent_coef`,
`seed`), prefixed `packer_ppo_*` / `physics_ppo_*`.

**Metrics** are logged continuously during training by
`training/callbacks.py::MlflowLoggingCallback`, one instance per round,
constructed with a `prefix` of `"packer"` or `"physics"` matching which
agent is training that round. Its `_on_rollout_end()` hook fires after every
SB3 rollout and pulls a fixed set of scalars out of SB3's internal logger:

```python
_TRACKED_KEYS = (
    "rollout/ep_rew_mean", "rollout/ep_len_mean",
    "train/loss", "train/entropy_loss", "train/value_loss", "train/approx_kl",
)
```

logged as `mlflow.log_metric(f"{prefix}/{key.split('/')[-1]}", value,
step=self.num_timesteps)` — so a single MLflow run ends up with, e.g.,
`packer/ep_rew_mean`, `physics/loss`, etc., all on a shared step axis, and
both agents' curves are visible side-by-side in `mlflow ui` without any
extra plumbing.

At the end of the run, two more things are logged: `mlflow.log_param(
"registered_version", version_id)` (linking the run to the registry entry
it produced) and `mlflow.set_tag("mlflow_run_id_for_version",
run.info.run_id)` (the reverse link, stored in registry-adjacent tooling if
needed).

## 2. The local JSON model registry

`mlops/registry.py::ModelRegistry` is, per its own docstring, "a
deliberately thin stand-in for a real model registry (e.g. the MLflow Model
Registry, or a cloud artifact store)" — everything else in `mlops/` talks
only to this class's interface (`add_version`, `get_version`,
`list_versions`, `update_metrics`, `get_champion`, `set_champion`), never to
the JSON file directly, so swapping in a real backend later only means
reimplementing this one class.

Storage is a single `registry.json` inside `registry_dir` (default
`models/registry`), shaped as `{"champion": <version_id or None>,
"versions": {<version_id>: <ModelVersion dict>}}`. The `ModelVersion`
dataclass schema:

```python
@dataclasses.dataclass
class ModelVersion:
    version: str              # e.g. "v3"
    created_at: str           # ISO-8601 timestamp
    packer_path: str          # path to packer.zip (SB3 PPO checkpoint)
    physics_path: str         # path to physics.zip
    train_config_hash: str    # sha256[:12] of (env_config, packer_config, physics_config)
    metrics: dict | None = None   # populated later by mlops/challenger.py's evaluation
```

`train.py` allocates the next id via `registry.next_version_id()` (`v1`,
`v2`, ... — parsed from existing `v<N>` keys), saves both SB3 checkpoints
into `registry.version_dir(version_id)`, computes `train_config_hash` via
`_config_hash(env_config, packer_config, physics_config)` (a
`json.dumps(..., sort_keys=True)` blob hashed with SHA-256, truncated to 12
hex chars — a fingerprint for "was this trained with the same configs"),
and calls `registry.add_version(version)`. **`metrics` is `None` at this
point** — a freshly trained version is registered but not yet evaluated;
that happens next, in the challenger gate.

The **champion pointer** (`index["champion"]`) is a single version id or
`None`, mutated only by `set_champion()`. There is no history of past
champions kept beyond what's implicit in each version's own record — the
registry only tracks "who is champion now."

The registry explicitly has "no file locking" and is "not safe for
concurrent writers" — it's designed for the local, single-process
train → evaluate → promote workflow this repo targets, not concurrent CI
runners.

## 3. Champion/Challenger promotion gate

`mlops/challenger.py::evaluate_challenger(challenger_id, registry,
env_config, eval_config, reward_shaping)` is the entry point:

1. Loads the challenger's `packer.zip`/`physics.zip` via `PPO.load(...)`.
2. Calls `_evaluate_version`, which runs `evaluation/benchmark.py::
   run_benchmark` **twice** — once with `randomization_mode="train"`, once
   with `"real_proxy"` — using `eval_config["num_eval_episodes"]` episodes
   starting at `eval_config["eval_seed_base"]`, and packages the result into
   an `EvalReport`:
   ```python
   @dataclasses.dataclass
   class EvalReport:
       version_id: str
       sim_utilization_pct: float
       real_proxy_utilization_pct: float
       sim2real_gap_pct: float
       stability_success_rate_pct: float
   ```
3. Writes that report back into the registry:
   `registry.update_metrics(challenger_id, dataclasses.asdict(challenger_report))`
   — this is what populates the previously-`None` `metrics` field.
4. Looks up the current champion. If there isn't one (or the champion *is*
   the challenger being evaluated), skips straight to the bootstrap path of
   `decide_promotion`. Otherwise, evaluates the champion the same way (a
   fresh `_evaluate_version` call — champion metrics are re-benchmarked on
   the same eval config every time, not read from a stale cached value) and
   calls `decide_promotion` with both reports.

### Gate conditions (`configs/challenger_eval.yaml`)

```yaml
num_eval_episodes: 20
eval_seed_base: 1000

promotion_gate:
  min_utilization_improvement_pct: 1.0   # challenger box-utilization must exceed champion by >= this many pp
  max_stability_regression_pct: 2.0      # challenger stability success rate may fall at most this many points below champion
  max_sim2real_gap_regression_pct: 3.0   # challenger's (sim - real_proxy) gap may widen by at most this many points vs champion's

bootstrap_auto_promote: true
```

`decide_promotion(challenger_report, champion_report, eval_config)` computes
three deltas and requires **all three** to pass:

| # | Delta | Formula | Default threshold | Fails if |
|---|---|---|---|---|
| 1 | Utilization improvement | `challenger.sim_utilization_pct - champion.sim_utilization_pct` | `min_utilization_improvement_pct = 1.0` | improvement `< 1.0` pp (i.e. the challenger must beat the champion's sim utilization by at least 1 percentage point — merely tying or marginally beating isn't enough) |
| 2 | Stability regression | `champion.stability_success_rate_pct - challenger.stability_success_rate_pct` | `max_stability_regression_pct = 2.0` | regression `> 2.0` pp (challenger's accept-that-actually-stayed-stable rate may drop by at most 2 points vs champion) |
| 3 | Sim2Real gap regression | `challenger.sim2real_gap_pct - champion.sim2real_gap_pct` | `max_sim2real_gap_regression_pct = 3.0` | gap widens by `> 3.0` pp (challenger's sim-vs-real_proxy overfitting signature may not get meaningfully worse than the champion's) |

Each failing condition appends a human-readable reason string (e.g.
`"Utilization improvement 0.42pp < required 1.0pp"`); if all three pass, a
single `"All promotion gate conditions satisfied."` reason is recorded
instead. The result is a `GateResult(promote: bool, reasons: list[str],
challenger_report, champion_report)`.

### Bootstrap case: no champion yet

If `champion_report is None` (first-ever version, or the registry is empty),
`decide_promotion` **skips all three numeric gate conditions entirely** and
just checks the config flag:

```python
if champion_report is None:
    promote = bool(eval_config.get("bootstrap_auto_promote", True))
    reasons = [
        "No champion registered yet -- bootstrap auto-promote."
        if promote
        else "No champion registered yet and bootstrap_auto_promote is disabled."
    ]
    return GateResult(promote=promote, reasons=reasons, challenger_report=challenger_report, champion_report=None)
```

With the shipped default `bootstrap_auto_promote: true`, the very first
evaluated version is always promoted unconditionally — there's no prior
champion to compare against, so the gate has nothing to hold it to. Setting
`bootstrap_auto_promote: false` would instead block even the first version
from auto-promoting until a human intervenes.

## 4. `mlops/deploy.py`: the CLI

`deploy.py`'s `main()` is the operational entry point tying registry +
gate together:

```
python -m marl_packing.mlops.deploy --challenger v3 [--auto-promote] \
    --registry-dir models/registry \
    --env-config configs/env.yaml \
    --config configs/challenger_eval.yaml \
    --physics-config configs/train_physics.yaml
```

It loads the registry and configs, calls `evaluate_challenger(...)` (the
full evaluate-both-modes-for-both-versions-and-decide pipeline above), and
prints a side-by-side comparison via `_print_report`:

```
Challenger v3: sim_utilization=78.20% real_proxy_utilization=70.10% sim2real_gap=8.10pp stability=96.50%
Champion   v2: sim_utilization=76.80% real_proxy_utilization=71.00% sim2real_gap=5.80pp stability=97.80%
Promotion gate: FAIL
  - Sim2Real gap regression 2.30pp > allowed 3.00pp   # (example only -- illustrates the reason format)
```

Then the promotion decision itself:

- If `result.promote` is `True` **and** `--auto-promote` was passed:
  `registry.set_champion(args.challenger)` — the champion pointer flips to
  the challenger's version id.
- If `result.promote` is `True` but `--auto-promote` was **not** passed:
  logs that the challenger passed the gate but leaves the champion
  unchanged (a dry-run / human-in-the-loop mode — you can inspect the
  report before committing).
- If `result.promote` is `False`: logs the failure and **`raise
  SystemExit(1)`** — the process exits non-zero, which is what makes this
  usable as a CI gate (a pipeline step can fail the build on a non-passing
  challenger without any extra parsing of stdout).

### What "deploy" means here

Per `deploy.py`'s own module docstring: this is explicitly "the local
stand-in for a 'deploy' step -- in a system with a real inference server,
promotion here is where you'd trigger pushing the new model to serving."
Concretely, `set_champion()` only rewrites one string
(`index["champion"]`) in `registry.json` — it does not build a container,
push an artifact to a model server, restart any service, or affect anything
outside the local registry directory. Wiring a real deployment target (e.g.
triggering a rollout against a serving endpoint keyed off
`registry.get_champion().packer_path` / `.physics_path`) is the natural next
step this module is a placeholder for.
