"""Vector-graph hybrid retrieval: return coherent *context clusters*, not
isolated top-k chunks.

Pipeline per query:
  1. VECTOR   — cosine search finds seed chunks (semantic entry points).
  2. GRAPH    — spreading activation from the seeds pulls in structurally
                related chunks (sequential neighbors, shared entities,
                paraphrases) that pure vector search misses.
  3. FUSE     — final score = alpha * vector_sim + (1 - alpha) * activation.
  4. CLUSTER  — surviving chunks are grouped into connected components of the
                graph and returned ordered by cluster relevance, so the
                caller injects whole coherent islands into the LLM context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from semantic_cache.chunker import Chunk
from semantic_cache.graph_store import GraphStore
from semantic_cache.vector_store import VectorStore


@dataclass
class ContextCluster:
    chunks: List[Chunk]          # ordered by (doc_id, position) for readability
    score: float                 # max fused score inside the cluster
    seed_ids: List[int]          # which members were direct vector hits

    def text(self, separator: str = "\n...\n") -> str:
        return separator.join(c.text for c in self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        alpha: float = 0.6,          # weight of the vector score in fusion
        seed_k: int = 5,
        hops: int = 2,
        decay: float = 0.5,
    ):
        self.vectors = vector_store
        self.graph = graph_store
        self.alpha = alpha
        self.seed_k = seed_k
        self.hops = hops
        self.decay = decay

    def retrieve(
        self,
        query: str,
        max_chunks: int = 12,
        max_clusters: Optional[int] = None,
    ) -> List[ContextCluster]:
        seeds = self.vectors.search(query, top_k=self.seed_k)
        if not seeds:
            return []
        seed_scores = {cid: max(sim, 0.0) for cid, sim in seeds}

        activation = self.graph.spread_activation(
            seed_scores, hops=self.hops, decay=self.decay
        )
        if activation:
            peak = max(activation.values()) or 1.0
            activation = {k: v / peak for k, v in activation.items()}

        fused: Dict[int, float] = {}
        for cid in set(seed_scores) | set(activation):
            if cid not in self.graph.nodes:
                continue
            fused[cid] = self.alpha * seed_scores.get(cid, 0.0) + (
                1 - self.alpha
            ) * activation.get(cid, 0.0)

        keep = sorted(fused, key=fused.get, reverse=True)[:max_chunks]
        clusters = self._group(keep, fused, set(seed_scores))
        return clusters[:max_clusters] if max_clusters else clusters

    def _group(
        self, kept: List[int], scores: Dict[int, float], seed_ids: set
    ) -> List[ContextCluster]:
        remaining = set(kept)
        clusters: List[ContextCluster] = []
        while remaining:
            start = max(remaining, key=lambda c: scores[c])
            members = self.graph.connected_component(start, remaining)
            remaining -= members
            chunks = sorted(
                (self.graph.nodes[m] for m in members),
                key=lambda c: (c.doc_id, c.position),
            )
            clusters.append(
                ContextCluster(
                    chunks=chunks,
                    score=max(scores[m] for m in members),
                    seed_ids=sorted(m for m in members if m in seed_ids),
                )
            )
        clusters.sort(key=lambda cl: cl.score, reverse=True)
        return clusters
