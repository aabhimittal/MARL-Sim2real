"""Graph side of the hybrid: chunks are nodes, edges encode why two chunks
belong in the same context cluster.

Edge types (weights are additive when types coincide):
  - sequential: adjacent chunks of the same document (narrative continuity)
  - entity:     chunks sharing extracted entities (topical coupling)
  - semantic:   embedding similarity above a threshold (paraphrase coupling)

Retrieval-time expansion uses spreading activation (a truncated personalized
PageRank): energy starts at the vector-search seeds and flows along weighted
edges, so a chunk with no lexical overlap with the query still surfaces when
the graph says it belongs to the same cluster.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

from semantic_cache.chunker import Chunk

EDGE_WEIGHTS = {"sequential": 1.0, "entity": 0.6, "semantic": 0.8}


class GraphStore:
    def __init__(self):
        self.nodes: Dict[int, Chunk] = {}
        # adjacency: node -> neighbor -> accumulated weight
        self._adj: Dict[int, Dict[int, float]] = defaultdict(dict)
        self._entity_index: Dict[str, Set[int]] = defaultdict(set)

    def __len__(self) -> int:
        return len(self.nodes)

    # ------------------------------------------------------------ mutation

    def add_chunk(self, chunk: Chunk, max_entity_fanout: int = 50) -> None:
        """Insert a chunk and wire its sequential + entity edges.

        `max_entity_fanout` skips edges through ubiquitous entities (an entity
        present in half the corpus carries no cluster signal, only noise).
        """
        self.nodes[chunk.chunk_id] = chunk

        for other in self.nodes.values():
            if (
                other.doc_id == chunk.doc_id
                and abs(other.position - chunk.position) == 1
                and other.chunk_id != chunk.chunk_id
            ):
                self._add_edge(chunk.chunk_id, other.chunk_id, "sequential")

        for entity in chunk.entities:
            peers = self._entity_index[entity]
            if len(peers) <= max_entity_fanout:
                for peer in peers:
                    self._add_edge(chunk.chunk_id, peer, "entity")
            peers.add(chunk.chunk_id)

    def add_semantic_edge(self, a: int, b: int, similarity: float) -> None:
        self._add_edge(a, b, "semantic", scale=similarity)

    def _add_edge(self, a: int, b: int, kind: str, scale: float = 1.0) -> None:
        if a == b or a not in self.nodes or b not in self.nodes:
            return
        w = EDGE_WEIGHTS[kind] * scale
        self._adj[a][b] = self._adj[a].get(b, 0.0) + w
        self._adj[b][a] = self._adj[b].get(a, 0.0) + w

    def remove_chunk(self, chunk_id: int) -> None:
        chunk = self.nodes.pop(chunk_id, None)
        if chunk is None:
            return
        for neighbor in list(self._adj[chunk_id]):
            self._adj[neighbor].pop(chunk_id, None)
        self._adj.pop(chunk_id, None)
        for entity in chunk.entities:
            self._entity_index[entity].discard(chunk_id)

    # ----------------------------------------------------------- retrieval

    def neighbors(self, chunk_id: int) -> Dict[int, float]:
        return dict(self._adj.get(chunk_id, {}))

    def spread_activation(
        self,
        seeds: Dict[int, float],
        hops: int = 2,
        decay: float = 0.5,
        min_energy: float = 1e-3,
    ) -> Dict[int, float]:
        """Propagate seed energy along weighted edges for `hops` rounds.

        Returns total energy per node — the graph's relevance score. Energy
        splits proportionally to edge weight and decays per hop, so close,
        strongly-linked chunks dominate distant ones.
        """
        energy = dict(seeds)
        frontier = dict(seeds)
        for _ in range(hops):
            next_frontier: Dict[int, float] = defaultdict(float)
            for node, e in frontier.items():
                nbrs = self._adj.get(node, {})
                total_w = sum(nbrs.values())
                if total_w == 0 or e < min_energy:
                    continue
                for nbr, w in nbrs.items():
                    next_frontier[nbr] += decay * e * (w / total_w)
            for node, e in next_frontier.items():
                energy[node] = energy.get(node, 0.0) + e
            frontier = dict(next_frontier)
        return energy

    def connected_component(self, start: int, allowed: Iterable[int]) -> Set[int]:
        """Component of `start` within the induced subgraph on `allowed`."""
        allowed = set(allowed)
        seen, stack = {start}, [start]
        while stack:
            node = stack.pop()
            for nbr in self._adj.get(node, {}):
                if nbr in allowed and nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        return seen
