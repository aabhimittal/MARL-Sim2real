"""SemanticCache: the user-facing facade tying chunker + vector store +
graph store + hybrid retriever into an ultra-long-context working memory.

Usage pattern with an LLM:
    cache = SemanticCache(max_chunks=5000)
    cache.ingest(long_document, doc_id="spec")        # as context streams in
    clusters = cache.query("how is stability judged") # before each model call
    prompt = cache.render(clusters)                   # inject only these

Instead of stuffing the whole history into the model window, only the
relevant context clusters are injected per call. Eviction is score-based:
chunks that are old, rarely retrieved, and weakly connected go first, so the
cache degrades gracefully as the conversation outgrows `max_chunks`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from semantic_cache.chunker import Chunk, chunk_text
from semantic_cache.graph_store import GraphStore
from semantic_cache.hybrid_retriever import ContextCluster, HybridRetriever
from semantic_cache.vector_store import Embedder, VectorStore


@dataclass
class ChunkMeta:
    inserted_at: float
    last_access: float
    hits: int = 0


@dataclass
class CacheStats:
    chunks: int = 0
    queries: int = 0
    evictions: int = 0
    ingested_docs: int = 0


class SemanticCache:
    def __init__(
        self,
        max_chunks: int = 5000,
        chunk_size: int = 120,
        overlap: int = 20,
        semantic_edge_threshold: float = 0.45,
        semantic_edge_k: int = 3,
        embedder: Optional[Embedder] = None,
        alpha: float = 0.6,
        clock=time.monotonic,
    ):
        self.max_chunks = max_chunks
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.semantic_edge_threshold = semantic_edge_threshold
        self.semantic_edge_k = semantic_edge_k
        self.vectors = VectorStore(embedder)
        self.graph = GraphStore()
        self.retriever = HybridRetriever(self.vectors, self.graph, alpha=alpha)
        self.meta: Dict[int, ChunkMeta] = {}
        self.stats = CacheStats()
        self._next_id = 0
        self._clock = clock

    # -------------------------------------------------------------- ingest

    def ingest(self, text: str, doc_id: Optional[str] = None) -> List[int]:
        """Chunk, embed, index, and graph-wire a document. Returns chunk ids."""
        doc_id = doc_id or f"doc{self.stats.ingested_docs}"
        chunks = chunk_text(
            text, doc_id=doc_id,
            chunk_size=self.chunk_size, overlap=self.overlap,
            start_id=self._next_id,
        )
        self._next_id += len(chunks)
        if hasattr(self.vectors.embedder, "fit_corpus"):
            self.vectors.embedder.fit_corpus([c.text for c in chunks])

        now = self._clock()
        for chunk in chunks:
            self._index_chunk(chunk, now)
        self.stats.ingested_docs += 1
        self.stats.chunks = len(self.graph)
        self._evict_if_needed()
        return [c.chunk_id for c in chunks]

    def _index_chunk(self, chunk: Chunk, now: float) -> None:
        # Wire semantic edges to its nearest existing neighbors BEFORE adding
        # itself to the vector index (so it doesn't match itself).
        neighbors = [
            (cid, sim)
            for cid, sim in self.vectors.search(chunk.text, top_k=self.semantic_edge_k)
            if sim >= self.semantic_edge_threshold
        ]
        self.vectors.add(chunk.chunk_id, chunk.text)
        self.graph.add_chunk(chunk)
        for cid, sim in neighbors:
            self.graph.add_semantic_edge(chunk.chunk_id, cid, sim)
        self.meta[chunk.chunk_id] = ChunkMeta(inserted_at=now, last_access=now)

    # --------------------------------------------------------------- query

    def query(self, question: str, max_chunks: int = 12) -> List[ContextCluster]:
        clusters = self.retriever.retrieve(question, max_chunks=max_chunks)
        now = self._clock()
        for cluster in clusters:
            for chunk in cluster.chunks:
                m = self.meta.get(chunk.chunk_id)
                if m:
                    m.hits += 1
                    m.last_access = now
        self.stats.queries += 1
        return clusters

    def render(self, clusters: List[ContextCluster], budget_chars: int = 8000) -> str:
        """Serialize clusters into a context block, best clusters first,
        truncated to a character budget (proxy for a token budget).
        """
        parts: List[str] = []
        used = 0
        for i, cluster in enumerate(clusters):
            block = f"[context cluster {i + 1} | score {cluster.score:.2f}]\n{cluster.text()}"
            if used + len(block) > budget_chars:
                if not parts:  # never return nothing: truncate the best cluster
                    parts.append(block[:budget_chars])
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)

    # ------------------------------------------------------------- evict

    def _eviction_score(self, chunk_id: int, now: float) -> float:
        """Lower score = evicted first. Blends recency, hit count, and graph
        degree (well-connected hub chunks are worth keeping: they glue
        clusters together).
        """
        m = self.meta[chunk_id]
        recency = 1.0 / (1.0 + (now - m.last_access))
        degree = len(self.graph.neighbors(chunk_id))
        return 2.0 * recency + 1.0 * m.hits + 0.2 * degree

    def _evict_if_needed(self) -> None:
        overflow = len(self.graph) - self.max_chunks
        if overflow <= 0:
            return
        now = self._clock()
        victims = sorted(self.meta, key=lambda cid: self._eviction_score(cid, now))
        for cid in victims[:overflow]:
            self.graph.remove_chunk(cid)
            self.vectors.remove(cid)
            self.meta.pop(cid, None)
            self.stats.evictions += 1
        self.stats.chunks = len(self.graph)
