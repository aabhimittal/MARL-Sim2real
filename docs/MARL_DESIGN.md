# MARL Design: The Packer/Physics Two-Agent System

This document explains the multi-agent design implemented in
`src/marl_packing/envs/packing_env.py`: why the environment is turn-based,
exactly how turns and rewards flow per item, and how the two agents are
trained with plain single-agent PPO instead of a native multi-agent trainer.

## The two agents

`PackingEnv` (a PettingZoo `AECEnv`, class in `envs/packing_env.py`) defines
exactly two agents, `possible_agents = ["packer", "physics"]`:

- **`packer`** looks at the current heightmap and the next item's dimensions
  and proposes where to put it. Its action space is
  `spaces.Box(low=0.0, high=1.0, shape=(3,))`: `(x_frac, y_frac, rot_frac)`,
  three numbers in `[0, 1]` that get decoded in `_apply_packer_action`:
  - `x_frac`/`y_frac` are scaled by the remaining free span
    (`bin_dims[0] - w`, `bin_dims[1] - d`) to get an absolute drop position.
  - `rot_frac` selects one of 6 axis-aligned rotations (see below).
- **`physics`** looks at the heightmap plus the packer's *specific* proposed
  placement (`proposed_item`: 6 floats — 3D center + 3D rotated dims) and
  chooses `spaces.Discrete(2)`: accept (1) or reject (0). Acceptance triggers
  a real PyBullet drop simulation (`envs/pybullet_sim.py::DropSimulator`)
  that determines whether the stack is actually stable.

### The 6 rotations

Box dimensions are sampled as `(x, y, z)` extents; a "rotation" is really an
index permutation of those three axes, applied to compute the placed
footprint `(w, d)` and height `h`. `packing_env.py` hardcodes all 6
permutations of 3 elements:

```python
_ROTATIONS = [
    (0, 1, 2), (1, 0, 2), (0, 2, 1),
    (2, 0, 1), (1, 2, 0), (2, 1, 0),
]
```

`rot_frac` selects an index via
`rot_id = min(int(rot_frac * len(_ROTATIONS)), len(_ROTATIONS) - 1)`, and
`w, d, h = item_dims[perm[0]], item_dims[perm[1]], item_dims[perm[2]]` picks
out the rotated extents used both for footprint placement and for the
PyBullet collision shape.

## Why AECEnv, not ParallelEnv

The module docstring states the reasoning directly, and it's a real
implementation constraint, not a style preference: **physics must observe
the packer's specific proposal before it can act.** `physics`'s observation
(`proposed_item`) *is* the packer's action for this item — there is no way
to compute a physics reward, or even a physics observation, without first
knowing exactly where and how the packer wants to place the box.

PettingZoo's `ParallelEnv` assumes all agents choose actions simultaneously
from the *same* observation, with no agent seeing another's this-step action
before acting. That doesn't fit here: this is a strictly sequential,
causally-dependent decision (propose, then judge the proposal), which is
exactly what turn-based `AECEnv` — with its explicit `agent_selection` and
per-agent `step()` calls — is built to express. Using `ParallelEnv` here
would force an awkward workaround (e.g., stashing the packer's action in
`infos` for physics to read one env-step later, which breaks the
single-step reward attribution described below).

## Turn/reward flow per item

Each item goes through exactly two `step()` calls, driven by PettingZoo's
`AgentSelector` (`self._agent_selector` in `reset()`/`step()`):

1. **Packer's turn.** `step(action)` with `agent_selection == "packer"` calls
   `_apply_packer_action`, which computes the candidate footprint, its base
   height from the current heightmap, and whether it fits under the bin's
   height ceiling (`fits_vertically`). This is stashed in
   `self._pending_placement` (a `_PendingPlacement` dataclass) — nothing is
   committed to the heightmap yet. Both agents' rewards are reset to `0.0` at
   the top of every `step()` call
   (`self.rewards = {a: 0.0 for a in self.agents}`), so **the packer's own
   turn always yields zero reward** — the packer doesn't learn anything about
   this item until physics resolves it next turn.
2. **Physics's turn.** `step(action)` with `agent_selection == "physics"`
   calls `_apply_physics_action`, which reads `self._pending_placement`,
   resolves the outcome (see below), and — critically — **sets both
   `self.rewards["packer"]` and `self.rewards["physics"]` in this single
   call**. This is the step where the item's fate (and both agents' credit
   assignment for it) is fully decided. `self._agent_selector.is_last()` is
   then true, so `_advance_item()` runs: it clears `_pending_placement`,
   increments `_item_idx`, and — if that was the last item — marks both
   agents terminated with an `episode_summary` (`utilization_pct`,
   `items_placed`, `items_total`) written into `infos`.

Turn order then wraps back to `packer` for the next item via
`self._agent_selector.next()`.

```mermaid
sequenceDiagram
    participant P as packer
    participant Env as PackingEnv
    participant Phys as physics
    participant Sim as DropSimulator (PyBullet)

    Env->>P: observe("packer") = {heightmap, next_item}
    P->>Env: step(x_frac, y_frac, rot_frac)
    Note over Env: _apply_packer_action stores _pending_placement<br/>reward = 0 for both agents this turn

    Env->>Phys: observe("physics") = {heightmap, proposed_item}
    Phys->>Env: step(accept | reject)
    alt invalid height
        Note over Env: no sim call — packer penalized, physics reward 0
    else accept
        Env->>Sim: simulate_drop(existing_boxes, candidate)
        Sim-->>Env: StabilityResult(stable?)
        alt stable
            Note over Env: commit placement; packer + physics both rewarded
        else unstable
            Note over Env: not committed; packer penalized, physics penalized
        end
    else reject
        Env->>Sim: simulate_drop(...) [ghost check, never committed]
        Sim-->>Env: StabilityResult(would have been stable?)
        Note over Env: packer penalized either way; physics reward depends<br/>on whether the reject was correct
    end
    Env->>Env: rewards["packer"], rewards["physics"] both set
    Note over Env: if last item, terminate + attach episode_summary
```

## The 5 outcomes and reward shaping

`_apply_physics_action` computes `item_frac_volume = prod(pp.dims) /
bin_volume` and branches into exactly 5 outcomes, each producing a
`(packer_reward, physics_reward)` pair. The scale constants and shaping
values are module-level in `packing_env.py`:

```python
_UTILIZATION_REWARD_SCALE = 10.0
_DISCARD_PENALTY_SCALE = 5.0
_INVALID_PLACEMENT_PENALTY_SCALE = 10.0

DEFAULT_REWARD_SHAPING = {
    "false_accept_penalty": -1.0,
    "false_reject_penalty": -0.3,
    "correct_decision_reward": 0.2,
}
```

`PackingEnv.__init__` merges any `reward_shaping` dict passed in over these
defaults; `training/train.py` passes `physics_config["reward_shaping"]` from
`configs/train_physics.yaml`, which currently just restates the same three
defaults explicitly:

```yaml
reward_shaping:
  false_accept_penalty: -1.0    # accepted a placement that then collapsed
  false_reject_penalty: -0.3    # rejected a placement that would have been stable
  correct_decision_reward: 0.2
```

| Outcome | Condition | `packer` reward | `physics` reward |
|---|---|---|---|
| `invalid_height` | proposal exceeds bin height (`fits_vertically == False`); no sim call at all | `-10.0 * item_frac_volume` | `0.0` |
| `accepted_stable` | physics accepts, `DropSimulator` reports `stable=True` | `+10.0 * item_frac_volume` | `+0.2` (`correct_decision_reward`) |
| `accepted_unstable` | physics accepts, sim reports `stable=False` | `-5.0 * item_frac_volume` | `-1.0` (`false_accept_penalty`) |
| `rejected_would_have_been_stable` | physics rejects; a private "ghost" sim run on the same candidate reports it *would* have been stable | `-5.0 * item_frac_volume` | `-0.3` (`false_reject_penalty`) |
| `rejected_correctly` | physics rejects; ghost sim confirms it would have been unstable | `-5.0 * item_frac_volume` | `+0.2` (`correct_decision_reward`) |

Notes on the shaping design:

- The packer's reward is always proportional to the item's fractional
  volume (`item_frac_volume`), so bigger boxes matter more — both as reward
  for successfully placing them and as penalty for having them discarded or
  rejected as too tall.
- Only `accepted_stable` actually commits the box (`_commit_placement`):
  updates the heightmap, appends to `_placed_boxes`, and adds to
  `_utilized_volume`. Every other outcome leaves the bin state unchanged —
  the item is effectively discarded for the rest of the episode (there's no
  retry queue).
- The "ghost" simulation on reject (`rejected_would_have_been_stable` /
  `rejected_correctly`) is never committed — it exists purely so `physics`
  gets a supervised-quality signal on whether its reject decision was
  actually correct, at the cost of one extra `simulate_drop` call per reject.

## Alternating independent-learner PPO

`training/train.py::train()` implements what its own docstring calls
"alternating independent-learner MARL": rather than a native multi-agent
trainer, each agent is trained with an ordinary single-agent
`stable_baselines3.PPO`, alternating which agent is being optimized while
the other's policy is frozen as the "opponent":

```python
for round_idx in range(n_rounds):
    train_packer_this_round = round_idx % 2 == 0
    if train_packer_this_round:
        # physics frozen (or RandomOpponent in round 0), packer trains
        ...
    else:
        # packer frozen, physics trains
        ...
```

`configs/train_packer.yaml`'s `schedule` block controls this: `n_rounds: 4`
(so packer trains in rounds 0 and 2, physics in rounds 1 and 3),
`timesteps_per_round: 4096` for full-scale runs, and
`smoke_timesteps_per_round: 256` for the fast CI/smoke path
(`train(..., full=False)`, used by `scripts/run_smoke_train.sh`).

This is a standard, simple MARL scheme — iterative best-response /
alternating self-play — chosen specifically because it lets both agents use
plain, off-the-shelf SB3 `PPO` with `MultiInputPolicy` (both observation
spaces are `spaces.Dict`), instead of requiring a heavier native
multi-agent RL library.

### How `single_agent_wrappers.py` makes this work

SB3's `PPO` expects a plain Gymnasium `Env`, but `PackingEnv` is a
two-agent PettingZoo `AECEnv`. `agents/single_agent_wrappers.py` bridges
this with two `gym.Env` wrappers, each presenting *one* agent's turn as an
ordinary single-agent step, and internally driving the other agent's turn
using a frozen "opponent" policy:

- **`PackerSingleAgentEnv`**: `step()` applies the packer's action to
  `base_env`, then immediately calls `physics_policy.predict(...)` and
  steps physics internally, so from SB3's point of view one `.step()` call
  = one full item resolved, with the reward being whatever
  `base_env.rewards["packer"]` came out to after physics resolved it.
- **`PhysicsSingleAgentEnv`**: symmetric — `_drive_packer_turn()` calls
  `packer_policy.predict(...)` to produce the proposal (at `reset()` and
  after every physics step), so SB3 only ever sees physics's own turns.

Both wrappers accept a `Policy` (a `predict(observation, deterministic)`
protocol matching SB3's own model interface) for the frozen opponent. Before
either agent has been trained (round 0), `RandomOpponent` — which just
samples the opponent's action space — stands in, so training can bootstrap
without a pretrained policy on either side. `train.py` swaps in the
just-trained `PPO` model instance as the opponent for the next round
(`physics_opponent = physics_model or RandomOpponent(...)`), so both models
gradually improve against increasingly competent counterparts.

Both agents share the *same* `base_env: PackingEnv` instance across rounds
(constructed once in `train()`), so PyBullet's `DropSimulator` client is
reused rather than rebuilt — only the wrapper and which model is "live"
changes between rounds.

At the end of all rounds, `train()` saves `packer.zip` and `physics.zip`
into a new version directory via `mlops/registry.py::ModelRegistry` (see
`docs/MLOPS.md` for what happens after registration).
