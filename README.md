# MARL-Sim2real

A **Multi-Agent Reinforcement Learning** system for 3D bin packing where a **Proposer Agent** picks box placements and orientations and a **Physics Agent** independently judges (and can veto) their stability — plus a **Sim2Real bridge** that carries the trained policies onto real hardware, and a **graph-based semantic cache** (vector-graph hybrid) that feeds an LLM only the relevant *context clusters* out of an ultra-long context.

Everything runs on **numpy alone** — no torch, no GPU, no API keys. PyBullet is an optional drop-in backend for higher-fidelity physics.

```
┌─────────────────────────  SIMULATION  ─────────────────────────┐
│                                                                │
│   ┌──────────────┐   propose (x, y, rot)   ┌───────────────┐   │
│   │   Proposer   │ ───────────────────────▶│  Physics      │   │
│   │   Agent      │                         │  Agent        │   │
│   │  (PPO policy)│ ◀───────────────────────│ (rules+critic │   │
│   └──────┬───────┘   verdict / veto        │  +PyBullet*)  │   │
│          │                                 └──────┬────────┘   │
│          ▼                 PackingEnv             │ online     │
│   reward = volume packed ± stability penalties ◀──┘ learning   │
│                                                                │
└──────┬────────────────── Sim2Real bridge ──────────┬───────────┘
       │  domain randomization (train-time)          │
       ▼                                             ▼
┌────────────────┐   paired outcomes   ┌────────────────────────┐
│  Real robot /  │ ───────────────────▶│  RealityGapCalibrator  │
│  MockRobot     │ ◀───────────────────│  widen randomization,  │
│  (digital twin │   safety-filtered   │  fine-tune the critic  │
│   resync/box)  │   greedy actions    └────────────────────────┘
└────────────────┘
```

## Repository layout

```
marl_packing/
  envs/packing_env.py          # voxel-grid 3D bin packing environment
  agents/proposer_agent.py     # masked-softmax placement policy (PPO)
  agents/physics_agent.py      # stability rules + learned critic + optional PyBullet
  agents/networks.py           # numpy MLP with manual backprop
  training/ppo_trainer.py      # joint MARL training loop (PPO-lite + GAE)
  sim2real/domain_randomization.py  # per-episode dynamics/observation perturbation
  sim2real/reality_gap.py      # gap measurement + adaptive randomization + critic finetune
  sim2real/bridge.py           # deployment loop: digital twin, safety filter, telemetry
  utils/geometry.py            # orientations, gravity drop, support polygon math
semantic_cache/
  chunker.py                   # overlapping chunks + lightweight entity extraction
  vector_store.py              # hash-embedding cosine index (pluggable embedder)
  graph_store.py               # sequential/entity/semantic edges + spreading activation
  hybrid_retriever.py          # vector seeds → graph expansion → context clusters
  cache_manager.py             # SemanticCache facade with score-based eviction
marl_sim2real/                 # second implementation track — see below
  env/                         # heightmap bin-packing env + parameterized stability engine
  agents/                      # REINFORCE proposer (top-K) + learned physics critic
  training/                    # co-training loop with physics replay buffer
  sim2real/                    # domain randomization, real-cell stub, CEM calibration, 5-stage bridge
  export/                      # .npz + SHA-256-verified manifest bundle for edge deployment
scripts/                       # train.py, evaluate.py, demo_cache.py, train_sim2real.py, run_bridge.py
tests/                         # pytest suite (runs in CI without torch/pybullet)
docs/ARCHITECTURE.md           # design decisions and data flow in depth
docs/MARL_SIM2REAL.md          # step-by-step guide to the marl_sim2real track
```

### Two implementation tracks

This repo hosts two complementary implementations of the same Proposer/Physics
MARL idea:

| | `marl_packing` (below) | `marl_sim2real` ([docs/MARL_SIM2REAL.md](docs/MARL_SIM2REAL.md)) |
|---|---|---|
| Policy training | PPO-lite + GAE, masked softmax | REINFORCE + baseline, top-K proposal/veto negotiation |
| Sim2Real | adaptive randomization + critic fine-tune | CEM system identification from real placement logs |
| Extras | vector-graph semantic cache for ultra-long contexts | hash-verified `.npz`+JSON edge export consumed by the **EdgePack** repo (registry / OTA rollout) |
| Entry points | `scripts/train.py`, `scripts/evaluate.py` | `scripts/train_sim2real.py`, `scripts/run_bridge.py` |

## Quick start

```bash
git clone https://github.com/aabhimittal/marl-sim2real
cd marl-sim2real
pip install -r requirements.txt        # numpy + pytest
pip install -e ".[physics]"            # optional: adds pybullet backend

python -m pytest tests/ -q             # verify the install

python scripts/train.py --episodes 200 --randomize   # train both agents
python scripts/evaluate.py                           # sim eval + sim2real deployment
python scripts/demo_cache.py                         # semantic cache walkthrough
```

---

# Step-by-step: how the system is built, end to end

## Step 1 — The packing environment (`marl_packing/envs/packing_env.py`)

The bin is a `(W, D, H)` voxel grid summarized as a 2D **heightmap** — the standard trick that makes 3D packing tractable: since boxes gravity-drop and never float, the top surface fully determines where the next box lands. An action is a flat index decoding to `(x, y, orientation)`; the orientation is one of the 6 axis-aligned rotations of the box (`utils/geometry.py`).

Key methods:
- `valid_action_mask()` — which placements physically fit (in bounds, under the rim). The policy only ever samples inside this mask.
- `preview(action)` — computes the drop height and support ratio **without committing**, which is what lets a second agent inspect a placement before it happens.
- `step(action, vetoed=, stable=)` — the multi-agent seam: the environment doesn't decide stability itself; the training loop passes in the Physics Agent's verdict, and the reward blends packed volume with stability penalties.

## Step 2 — The Proposer Agent (`agents/proposer_agent.py`)

A masked-softmax policy over all `W × D × 6` placements. Observation = normalized heightmap + a lookahead window of upcoming box dimensions + boxes remaining. Policy and value heads are two small numpy MLPs (`agents/networks.py`) with manual backprop — the point is a fully transparent, dependency-free training loop; every module mirrors the torch API (`forward`/`backward`/`state_dict`) so upgrading to torch is mechanical.

## Step 3 — The Physics Agent (`agents/physics_agent.py`)

Three layers, cheapest first:
1. **Analytic rules** — support ratio ≥ 30%, center of mass over the support region, slenderness limit for tall thin boxes. Fast necessary conditions.
2. **Learned critic** — an MLP over 7 physics features, trained *online* against the settle-test ground truth every single placement (logistic-regression loss, `learn()`). This is the transferable part: the same `learn()` later fine-tunes on **real robot outcomes**.
3. **PyBullet settle test** *(optional)* — rebuilds the pile from the heightmap, drops the box, measures drift and tilt after 240 steps. When pybullet isn't installed, a deterministic analytic settle model stands in, so CI never needs the dependency.

`judge()` returns a `StabilityVerdict` with a **veto bit**: vetoed boxes go back to the depot and the proposer eats a penalty — the adversarial-cooperative pressure that teaches the proposer to *propose stable placements in the first place*.

## Step 4 — Joint MARL training (`training/ppo_trainer.py`)

Per placement: proposer samples → physics agent judges the previewed geometry → environment commits or rejects → **both agents learn**: the physics critic updates online (BCE against its simulation backend), and the proposer collects a PPO rollout (GAE advantages, clipped surrogate, minibatch epochs, entropy bonus — all hand-derived on the numpy MLPs). Watch the two learning curves in the training log: `physics BCE` falls as the critic calibrates, `fill` climbs as the proposer packs denser without triggering vetoes.

## Step 5 — Domain randomization (`sim2real/domain_randomization.py`)

Reality never matches the simulator, so we train against a *distribution* of simulators. `DomainRandomizer` wraps the env without modifying it and perturbs, per episode: box dimension tolerance (manufacturing error), placement jitter (gripper error), heightmap noise (depth-camera error), and stability-threshold shifts (friction variation). A policy that packs well across all of these has learned strategies that don't depend on any one simulator's quirks — that's the Sim2Real bet.

## Step 6 — Measuring and closing the reality gap (`sim2real/reality_gap.py`)

Deployment produces paired outcomes: *what the sim predicted* vs *what the robot saw*. `RealityGapCalibrator.report()` quantifies the gap (disagreement rate, critic calibration error) and does two things with it:
- **Adaptive domain randomization** — `suggested_config` widens the randomization ranges proportionally to the measured gap (capped at 3×), so retraining covers the regime reality actually lives in.
- **Critic fine-tuning** — `finetune_physics_agent()` re-trains the stability critic on the real outcomes. The proposer's policy usually transfers as-is; it's the *stability model* that needs grounding in reality.

## Step 7 — The deployment bridge (`sim2real/bridge.py`)

`Sim2RealBridge` runs the trained agents against a `RobotInterface` (implement it over your ROS/vendor SDK; `MockRobot` implements it over a deliberately-mismatched simulator so the whole loop is testable without hardware). Per box:
1. read the **real** heightmap from the depth camera and resync the digital twin — closing the loop every placement so sim drift never accumulates;
2. rank actions by policy probability, take the best one the Physics Agent doesn't veto (a safety filter, walking down the ranking up to `max_retries`);
3. execute, then feed the real outcome into the calibrator.

Run `scripts/evaluate.py` to watch the full cycle: deploy → measure gap → fine-tune critic → redeploy with a smaller gap.

## Step 8 — The semantic cache (`semantic_cache/`)

Ultra-long contexts (multi-document corpora, long agent transcripts) don't fit in a model window, and flat top-k vector retrieval returns *isolated* chunks — it misses the sequel paragraph, the definition three pages earlier, the paraphrase that shares no keywords. The fix is a **vector-graph hybrid**:

1. **Chunking** (`chunker.py`) — overlapping word-window chunks + cheap entity extraction (CamelCase/dotted/rare terms) used to build edges.
2. **Vector store** (`vector_store.py`) — deterministic hash-based TF-IDF embeddings behind an `Embedder` protocol (swap in sentence-transformers or an API in one line), flat cosine index.
3. **Graph store** (`graph_store.py`) — nodes are chunks; weighted edges say *why* two chunks belong together: `sequential` (adjacent in the same doc), `entity` (shared entities), `semantic` (embedding similarity above threshold).
4. **Hybrid retrieval** (`hybrid_retriever.py`) — vector search finds seed chunks; **spreading activation** (truncated personalized PageRank) flows energy from the seeds along edges for 2 hops; scores fuse as `α·cosine + (1−α)·activation`; survivors are grouped into **connected components** and returned as ordered `ContextCluster`s — whole coherent islands, not fragments.
5. **Cache manager** (`cache_manager.py`) — `SemanticCache.ingest()` as context streams in, `query()` before each model call, `render()` to serialize the best clusters into a character/token budget. When the cache outgrows `max_chunks`, **score-based eviction** removes chunks that are old, rarely retrieved, *and* weakly connected — hub chunks that glue clusters together are kept.

```python
from semantic_cache import SemanticCache

cache = SemanticCache(max_chunks=5000)
cache.ingest(open("spec.md").read(), doc_id="spec")
cache.ingest(transcript, doc_id="chat")

clusters = cache.query("how is placement stability judged?")
prompt_context = cache.render(clusters, budget_chars=8000)  # inject only this
```

## Step 9 — Verify everything

```bash
python -m pytest tests/ -q
```

The suite covers geometry, environment mechanics, both agents, the trainer, the full sim2real loop against `MockRobot`, and the cache (retrieval, clustering, eviction) — all seeded and CPU-only.

---

## Upgrade paths

| Component | Ships as | Production swap |
|---|---|---|
| Policy/critic nets | numpy MLP, manual backprop | torch modules (API mirrors `forward`/`state_dict`) |
| Physics ground truth | analytic settle model | `pip install pybullet`, `PhysicsAgent(use_pybullet=True)` |
| Embeddings | hash-based TF-IDF | any `Embedder` (sentence-transformers, embeddings API) |
| Vector index | flat cosine (exact) | FAISS/HNSW behind the same `add/search/remove` |
| Robot | `MockRobot` | implement `RobotInterface` over ROS / vendor SDK |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for design rationale, data flow, and the reward/feature definitions in detail.

## License

MIT — see [LICENSE](LICENSE).
