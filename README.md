# MARL-Sim2real

Multi-Agent Reinforcement Learning for 3D bin packing with physics-validated
stability and Sim2Real transfer. The repository contains **two complementary
tracks** plus a standalone semantic-cache module:

| Track | Packages | Stack | Focus |
|---|---|---|---|
| **A — Drift-correction pipeline** | `marl_sim2real/` | torch + PyBullet + networkx | MARL packing → PyBullet ideal traffic data → dynamic GNN path optimization → k-sigma drift detection → GNN self-correction → complexity-routed LLM calls |
| **B — MLOps training loop** | `src/marl_packing/` | SB3 + PettingZoo + PyBullet + MLflow | Packer/Physics agents trained with alternating independent-learner PPO, domain-randomized real-world proxy benchmark, MLflow tracking + model registry with a Champion/Challenger promotion gate |
| — | `semantic_cache/` | numpy | vector-graph hybrid retrieval: feeds long-context LLM calls only the relevant *context clusters* |

Both tracks run headless on CPU and share one CI (`pytest tests -q`).

---

# Track A — `marl_sim2real`: the drift-correction pipeline

Four systems chained together:

1. **Multi-Agent RL packing** — a *Proposer Agent* learns to propose 3D packing
   orientations; a *Physics Agent* validates every proposal by dropping the item
   in a **PyBullet** simulation and measuring stability. They cooperate through a
   shared reward.
2. **Dynamic GNN path optimization** — a graph neural network re-embeds a
   warehouse waypoint graph on every tick and predicts per-edge traversal
   latency; Dijkstra plans robot routes over the predicted weights in real time.
3. **Sim2Real drift detection** — PyBullet generates *ideal* traffic data
   (per-edge latency distributions). Live robot latency is compared against this
   baseline; a deviation beyond **k standard deviations** triggers a
   **self-correction** script that recalibrates the GNN's edge weights.
4. **Token-cost optimization** — a complexity router scores each LLM task and
   switches between small/medium/large Claude models, so routine narration never
   pays large-model prices.

```mermaid
flowchart LR
    subgraph MARL["1 · MARL Packing"]
        P[Proposer Agent<br/>policy gradient] -->|orientation + position| Q[Physics Agent<br/>PyBullet drop test]
        Q -->|stability score / reject| P
    end
    subgraph SIM["2 · Sim2Real Bridge"]
        B[PyBullet ideal<br/>traffic generator] -->|baseline μ, σ per edge| D[Drift Detector<br/>k-sigma rule]
        R[Real robot latency] --> D
        D -->|DriftEvent| C[Self-Correction<br/>recalibrate edge biases]
    end
    subgraph GNN["3 · Path Optimization"]
        G[Dynamic GNN<br/>edge latency] --> O[Dijkstra<br/>route planner]
    end
    B -->|training data| G
    C -->|updated edge weights| G
    D -->|event severity| L[4 · Complexity Router<br/>Haiku / Sonnet / Opus]
```

## Quickstart (Track A)

```bash
pip install -r requirements.txt && pip install -e .
python -m pytest tests/ -q       # full test suite, both tracks

python scripts/run_pipeline.py   # compact end-to-end demo of every Track A stage
```

Stage-by-stage (full-size runs):

```bash
python scripts/train_marl.py --episodes 200        # 1. train the packing agents
python scripts/generate_sim_data.py --runs 30      # 2. PyBullet baseline + GNN pre-train
python scripts/monitor_drift.py --ticks 120 --k 3  # 3. live drift monitoring + self-correction
```

Everything degrades gracefully: without PyBullet a heuristic physics fallback is
used, and without `ANTHROPIC_API_KEY` the router returns deterministic offline
stubs — so the whole pipeline runs in CI. Track A knobs live in
`configs/default.yaml`, mirroring `marl_sim2real/config.py`.

## Step-by-step implementation guide (Track A)

This section walks through how Track A is built, in the order you would build
it yourself.

### Step 1 — The packing environment (`marl_sim2real/envs/packing_env.py`)

The bin is a `W×D` **heightmap**: each cell stores the current stack height.
This turns 3D packing into a 2.5D problem that is cheap to simulate and easy
to featurize.

- An **item** is a box `(w, d, h)` in grid cells. An **orientation** is one of
  the 6 axis permutations of those dims.
- An **action** is a single integer decoding to `(orientation, x, y)`.
  The resting height `z` is computed by gravity: the max heightmap value under
  the item's footprint.
- The **observation** is the normalized heightmap plus the current item's
  dimensions — everything the proposer needs to decide where the item fits.
- `support_ratio()` measures what fraction of the footprint is actually
  supported — used by the heuristic physics fallback.

### Step 2 — The Proposer Agent (`marl_sim2real/agents/proposer_agent.py`)

A REINFORCE policy-gradient agent with a learned value baseline:

- A small MLP maps the observation to logits over all
  `6 × W × D` actions plus a state-value estimate.
- **Feasibility masking**: geometrically impossible placements (out of bounds,
  over height) get `-inf` logits, so the agent only samples valid candidates
  and never wastes physics simulations on nonsense.
- After each episode, discounted returns are normalized, advantages are
  computed against the value baseline, and one clipped gradient step updates
  the policy (entropy bonus keeps exploration alive).

### Step 3 — The Physics Agent (`marl_sim2real/agents/physics_agent.py`)

The second agent doesn't learn — it *judges*. For each proposal it:

1. Builds a headless PyBullet scene (`p.DIRECT`): floor plane + all previously
   committed items as static boxes.
2. Spawns the candidate box 5 mm above its proposed pose and steps the
   simulation ~120 steps so it settles under gravity.
3. Measures **horizontal displacement** and **tilt** (angle between the body's
   local z-axis and world z, recovered from the rotation matrix).
4. Verdict: stable iff displacement ≤ 2 cm and tilt ≤ 10°; also returns a
   smooth score in `[0, 1]` used as shaped reward.

An analytic fallback (support ratio + aspect ratio) keeps everything runnable
where PyBullet isn't available.

### Step 4 — MARL coordination (`marl_sim2real/agents/coordinator.py`)

The cooperative protocol per item:

| Physics verdict | Environment effect | Proposer reward |
|---|---|---|
| stable | placement committed | `10 × volume_gain + 0.5 × stability_score` |
| unstable | item rejected | `−0.5` penalty |

Both agents optimize the same objective (dense *and* stable packing), which
makes this cooperative MARL: the proposer gradually internalizes the physics
agent's judgment and stops proposing configurations that will be rejected.

### Step 5 — Warehouse graph + Dynamic GNN (`marl_sim2real/gnn/`)

- `WarehouseGraph`: a random-geometric graph of waypoints with dynamic node
  features (queue length, robot density) and edge features (distance, load,
  surface type) resampled every tick.
- `DynamicGNN`: plain-torch message passing (no torch_geometric). Each layer
  computes edge messages from `[h_src, h_dst, edge_attr]`, aggregates by
  destination, and residually updates node embeddings. A softplus head predicts
  a **positive latency per edge**. "Dynamic" = same weights, time-varying
  inputs: the graph is re-embedded on every planning call.
- Crucially, the GNN carries a **per-edge calibration bias** parameter —
  a `num_edges`-sized knob that self-correction can tune *without touching the
  shared network weights*.
- `PathOptimizer` converts predicted latencies into a weighted digraph and runs
  Dijkstra — routes shift automatically as congestion changes.

### Step 6 — Ideal data + k-sigma drift detection (`marl_sim2real/sim2real/`)

**Ideal data** (`ideal_data_generator.py`): PyBullet simulates a
velocity-controlled robot traversal (capturing acceleration ramps and friction
losses), scaled per edge by distance and slowed by congestion. Repeating this
over sampled snapshots yields per-edge latency distributions:
the **sim baseline** `(μ_e, σ_e)` — plus (snapshot, latency) pairs that
pre-train the GNN.

**Detection** (`drift_detector.py`): each edge keeps a rolling window of real
observations, and drift triggers when

```
|mean(real_window_e) − μ_e| > k · σ_e        (default k = 3)
```

Windowed means (rather than single samples) reject one-off sensor spikes while
still reacting within one window length. The emitted `DriftEvent` carries edge
ids, z-scores, and observed vs expected means.

**Self-correction** (`self_correction.py`): two coordinated mechanisms fire on
every event —

1. **Baseline EMA** — pull `μ_e` toward observed reality (and widen `σ_e`) so
   the detector converges instead of re-firing forever on a known gap.
2. **Targeted bias fine-tuning** — freeze the shared GNN, mask the loss to the
   drifted edges only, and take a few gradient steps on the per-edge biases
   against recent real observations. Planning immediately reflects corrected
   latencies; the rest of the network is untouched, so a handful of local
   measurements can't destabilize global predictions.

Each correction is logged (JSONL) with pre/post prediction error on the
drifted edges — the demo shows error dropping and z-scores decaying across
successive events (e.g. 11.9σ → 8.8σ → 5.1σ).

### Step 7 — Token optimization by model switching (`marl_sim2real/llm/`)

Operational LLM tasks (narrating drift events, diagnosing failures) vary
hugely in difficulty. The router computes a complexity score in `[0, 1]` from
cheap signals:

| Signal | Weight | Intuition |
|---|---|---|
| prompt length (saturating) | 0.15 | more context to digest |
| drift severity above 3σ | 0.30 | 3σ is routine, 8σ+ is an anomaly |
| affected-edge blast radius | 0.20 | multi-edge failures need cross-edge reasoning |
| caller-flagged reasoning | 0.35 | multi-step logic required |
| prior failed attempts | +0.35 each | escalate on failure |

and thresholds it into tiers: `< 0.33 →` **Haiku**, `< 0.66 →` **Sonnet**,
else **Opus**. Every decision is recorded; `savings_report()` compares realized
cost against always-using-the-largest-model (the demo shows ~47–93% savings
depending on the event mix). With `ANTHROPIC_API_KEY` set, calls go to the real
API; otherwise a deterministic offline stub keeps the pipeline self-contained.

---

# Track B — `src/marl_packing`: the MLOps training loop

A **packer** agent proposes box positions and orientations, and a **physics**
agent accepts or rejects each proposal, with PyBullet drop simulation as the
ground truth for stack stability. The two agents are trained with alternating
independent-learner PPO (Stable-Baselines3) on a PettingZoo `AECEnv`. Sim2Real
transfer is measured against a domain-randomized "real-world proxy" benchmark,
and an MLOps loop (MLflow tracking + a local JSON model registry) gates
deployment through a Champion/Challenger promotion gate.

## Quickstart (Track B)

```bash
pip install -r requirements.txt && pip install -e .   # or: make setup

make test             # run the shared test suite
make smoke-train      # fast CPU smoke run: env -> agents -> PPO -> MLflow -> registry
make challenger-eval  # run a challenger through the promotion gate
make mlflow-ui        # browse MLflow runs
```

Other targets: `make lint` (ruff), `make train` (full-scale timesteps),
`make promote CHALLENGER=<version_id>`, `make clean`.

## Track B documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — system overview and how the pieces fit together
- [docs/MARL_DESIGN.md](docs/MARL_DESIGN.md) — the two-agent design and alternating training scheme
- [docs/SIM2REAL.md](docs/SIM2REAL.md) — domain randomization and the real-world proxy benchmark
- [docs/MLOPS.md](docs/MLOPS.md) — MLflow tracking, model registry, and the Champion/Challenger gate

## Scope and limitations (Track B)

- Training runs are deliberately short CPU smoke runs
  (`smoke_timesteps_per_round: 256`) that verify the pipeline end-to-end; no
  model quality claims are made. `make train` runs the full-scale schedule if
  you have the compute.
- There is no physical robot or camera rig. The "real-world" benchmark is a
  domain-randomization proxy — the same simulator with wider
  friction/mass/restitution ranges and pose noise (`real_proxy` in
  `configs/env.yaml`) — and
  `src/marl_packing/evaluation/real_world_reference.csv` is an illustrative
  placeholder, not genuine field data. See [docs/SIM2REAL.md](docs/SIM2REAL.md).

---

# Also in this repository: a semantic cache module

`semantic_cache/` is an unrelated, separately-developed feature: a vector-graph
hybrid retrieval cache for feeding long-context LLM calls only the relevant
*context clusters* out of an ultra-long context (chunking + a cosine vector
index + a chunk graph with spreading-activation retrieval + score-based
eviction). It has no dependency on the packing/MARL code. See
`semantic_cache/cache_manager.py` for the `SemanticCache` facade and
`scripts/demo_cache.py` for a runnable walkthrough; its tests live in
`tests/test_cache.py`.

---

## Repository layout

```
marl_sim2real/            Track A: config, packing env, REINFORCE proposer, PyBullet
                          judge, MARL coordinator, dynamic GNN + Dijkstra optimizer,
                          ideal-data generator, k-sigma drift detector, self-correction,
                          complexity router
src/marl_packing/         Track B: PettingZoo AECEnv, PyBullet drop simulator, domain
                          randomization, SB3 wrappers, alternating PPO training loop,
                          paired-seed benchmark, MLflow + registry + challenger gate
semantic_cache/           vector-graph hybrid retrieval cache
scripts/                  Track A: train_marl, generate_sim_data, monitor_drift,
                          run_pipeline · Track B: run_smoke_train.sh,
                          run_challenger_eval.sh, visualize_packing · demo_cache
configs/                  default.yaml (Track A) · env.yaml, train_packer.yaml,
                          train_physics.yaml, challenger_eval.yaml (Track B)
docs/                     Track B design documents
models/registry/          local JSON model registry (weights untracked)
tests/                    combined pytest suite for both tracks + semantic cache
```

## License

MIT — see [LICENSE](LICENSE).
