# Architecture

Design rationale and data flow for the three subsystems. Read the README's
step-by-step guide first; this document goes one level deeper.

## 1. MARL packing

### Why two agents instead of one policy with a stability reward?

A single policy that maximizes `volume - instability_penalty` learns whatever
the *training* simulator considers stable — the stability model is baked
irreversibly into the policy weights. Splitting the roles keeps the concerns
separable:

- The **Proposer** learns *packing strategy* (where density comes from),
  which transfers across physics regimes largely unchanged.
- The **Physics Agent** owns the *stability model* — the part that is wrong
  on real hardware — as a small, independently retrainable critic. Closing
  the reality gap then means fine-tuning a 7-feature logistic model on a few
  hundred real outcomes instead of re-running RL.

At deployment the Physics Agent doubles as a **runtime safety filter**: it
screens each proposed action and the bridge falls through to the next-best
action on a veto. A monolithic policy has no such interception point.

### Environment mechanics

- State is a heightmap `(W, D)` because gravity-dropped, axis-aligned boxes
  make the top surface a sufficient statistic of the pile.
- `step()` deliberately takes `vetoed`/`stable` as *inputs*: the env is the
  arbiter of geometry, never of physics. That inversion is what makes both
  the second agent and the sim2real perturbation wrappers composable.
- Rewards: `+volume/bin_volume * scale` per placed box; `penalty_invalid`
  for out-of-bounds proposals; `penalty_veto` when the Physics Agent rejects;
  `penalty_unstable` when a committed placement collapses in the settle test.

### Training loop invariants

- The physics critic trains **every placement** (online logistic regression,
  gradient `p − y`), so it needs no separate dataset or schedule.
- The proposer trains **every episode** with PPO-lite: GAE(λ) advantages,
  clipped surrogate (gradient zeroed where the ratio is clipped), entropy
  bonus on the masked softmax, minibatched epochs.
- All gradients flow through the hand-written `MLP.backward`; there is no
  autograd anywhere, which keeps the dependency surface at exactly `numpy`.

## 2. Sim2Real

The bridge follows the standard three-phase recipe, each phase one module:

| Phase | Module | Mechanism |
|---|---|---|
| Robustify in sim | `domain_randomization.py` | per-episode perturbation of dims, actions, observations, thresholds |
| Measure the gap | `reality_gap.py` | paired sim-prediction vs real-outcome records → disagreement & calibration metrics |
| Close the gap | `reality_gap.py` + `bridge.py` | widen randomization ∝ gap (adaptive DR), fine-tune critic on real outcomes, resync digital twin every placement |

Two details worth calling out:

- **Per-placement twin resync.** The bridge never trusts its own forward
  model between placements; it reads the robot's depth camera and overwrites
  the twin's heightmap before every decision. Sim2Real failures compound
  through drift — resyncing bounds the error to a single placement.
- **`MockRobot` is intentionally miscalibrated** (stricter support threshold,
  sensor noise) so the shipped tests exercise a *nonzero* reality gap and the
  calibrator has something real to measure. Swapping in hardware means
  implementing the 3-method `RobotInterface` and nothing else.

## 3. Semantic cache (vector-graph hybrid)

### The retrieval problem it solves

Flat top-k vector search over a long context returns the k chunks most
similar to the query *individually*. It systematically misses:

- the **next/previous chunk** of a hit (narrative continuity),
- chunks that share **entities** with a hit but use different vocabulary,
- **paraphrases** of a hit ranked just below the cutoff.

These are exactly the three edge types in `GraphStore` (`sequential`,
`entity`, `semantic`). The retrieval contract changes from "k most similar
chunks" to "the *context clusters* the query touches."

### Scoring pipeline

```
query ──▶ vector search ──▶ seeds {id: cosine}
                │
                ▼
      spread_activation(seeds, hops=2, decay=0.5)
      energy splits along edges ∝ weight, decays per hop
                │
                ▼
      fused = α·cosine + (1−α)·normalized_activation
                │
                ▼
      top-N by fused score ──▶ connected components ──▶ ordered clusters
```

Spreading activation is a truncated personalized PageRank: two hops covers
"neighbor of a neighbor" (e.g. the paragraph after a paraphrase of the
query) while staying O(edges touched) — no global iteration, no matrix
inversion, insertions stay cheap.

### Cache behavior at scale

`SemanticCache` is a working memory, not a database. When `max_chunks` is
exceeded, eviction removes the chunks minimizing
`2·recency + hits + 0.2·graph_degree` — old, never-retrieved, weakly-linked
chunks go first. Degree is in the score on purpose: hub chunks knit clusters
together, and evicting them fragments retrieval for every neighbor.

Complexities (n = cached chunks, per operation):

| Operation | Cost |
|---|---|
| ingest per chunk | O(n) vector scan for semantic edges + O(entities) wiring |
| query | O(n) exact cosine + O(edges in 2-hop ball) activation |
| evict | O(n log n) on overflow only |

The flat exact index is the right default below ~10⁵ chunks; beyond that,
`VectorStore` is a 3-method seam (`add/search/remove`) for FAISS/HNSW, and
`HashingEmbedder` a 1-method seam (`embed`) for real embedding models.

### Using it with an LLM

The cache is model-agnostic: `render()` emits a plain-text context block cut
to a budget. The intended loop —

```
while conversation continues:
    cache.ingest(new_turns / new_documents)
    clusters = cache.query(current_question)
    call_llm(system + cache.render(clusters, budget) + question)
```

— keeps per-call context near-constant while the underlying corpus grows
without bound.
