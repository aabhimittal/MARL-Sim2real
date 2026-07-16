#!/usr/bin/env python
"""Walkthrough of the vector-graph hybrid semantic cache.

Ingests three documents (two of which share entities so graph expansion is
visible), runs queries, and shows for each retrieved cluster which chunks
were direct vector hits (seeds) vs pulled in by spreading activation.

    python scripts/demo_cache.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_cache import SemanticCache

DOC_PHYSICS = """
The PhysicsAgent judges every placement the ProposerAgent suggests. It computes
a support ratio over the footprint, verifies the center of mass sits above the
support region, and applies a slenderness limit for tall thin boxes. A learned
critic network scores the same placement features and produces a confidence.
When both the analytic rules and the critic distrust a placement, the
PhysicsAgent issues a veto and the box is returned to the depot. The veto
threshold defaults to a confidence of 0.25 and is tunable per deployment.
An optional PyBullet backend replaces the analytic settle model: it rebuilds
the pile from the heightmap, drops the box, and measures drift and tilt.
""".strip()

DOC_BRIDGE = """
The Sim2RealBridge deploys trained policies against a RobotInterface. Before
every decision it reads the depth camera heightmap and resynchronizes its
digital twin, so simulator drift never accumulates across placements. The
PhysicsAgent acts as a runtime safety filter: vetoed actions are skipped and
the next-best action by policy probability is tried instead. Every executed
placement produces a paired outcome that the RealityGapCalibrator records.
The calibrator widens domain randomization ranges in proportion to the
measured disagreement rate and fine-tunes the critic on real outcomes.
""".strip()

DOC_RECIPES = """
To make a classic sourdough loaf, feed the starter twelve hours before mixing.
Combine flour, water, and salt, then rest the dough for an hour. Perform four
sets of stretch and folds at thirty minute intervals. Bulk fermentation takes
four to six hours at room temperature. Shape the loaf, proof it overnight in
the refrigerator, and bake in a preheated dutch oven for twenty minutes with
the lid on and twenty five minutes with the lid off until deeply browned.
""".strip()

QUERIES = [
    "how does the physics agent veto an unstable placement",
    "what keeps the digital twin synchronized with the real robot",
    "how long should bulk fermentation take",
]


def main() -> None:
    cache = SemanticCache(max_chunks=100, chunk_size=40, overlap=8)
    for doc_id, text in [("physics", DOC_PHYSICS), ("bridge", DOC_BRIDGE), ("recipes", DOC_RECIPES)]:
        ids = cache.ingest(text, doc_id=doc_id)
        print(f"ingested {doc_id!r}: {len(ids)} chunks")

    for query in QUERIES:
        print("\n" + "=" * 72)
        print(f"QUERY: {query}")
        clusters = cache.query(query, max_chunks=8)
        print(f"-> {len(clusters)} cluster(s)")
        for i, cluster in enumerate(clusters):
            seeds = set(cluster.seed_ids)
            expanded = [c.chunk_id for c in cluster.chunks if c.chunk_id not in seeds]
            docs = sorted({c.doc_id for c in cluster.chunks})
            print(f"  cluster {i + 1}: score {cluster.score:.3f} | {len(cluster)} chunks "
                  f"| docs {docs} | vector seeds {sorted(seeds)} | graph-expanded {expanded}")
        print("\nrendered context (600-char budget):")
        print(cache.render(clusters, budget_chars=600))

    print("\n" + "=" * 72)
    print("cache stats:", cache.stats)

    # ---- eviction demo -----------------------------------------------------
    print("\nEviction demo: shrinking the cache to 6 chunks and flooding it...")
    small = SemanticCache(max_chunks=6, chunk_size=20, overlap=4)
    small.ingest(DOC_PHYSICS, doc_id="physics")
    small.query("physics agent veto confidence")  # touch: protects these chunks
    small.ingest(DOC_RECIPES, doc_id="recipes")
    print("stats after overflow:", small.stats)
    survivors = sorted({c.doc_id for c in small.graph.nodes.values()})
    print("surviving docs:", survivors, "(recently-queried, well-connected chunks are kept)")


if __name__ == "__main__":
    main()
