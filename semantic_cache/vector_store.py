"""Vector side of the hybrid: embeddings + cosine similarity search.

`HashingEmbedder` is a deterministic, dependency-free TF-IDF-weighted feature
hasher — no model download, no API key, fully reproducible. The `Embedder`
protocol is the seam: point it at sentence-transformers or an embeddings API
for production quality without touching the retriever.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Protocol, Sequence, Tuple

import numpy as np

from semantic_cache.chunker import tokenize


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Feature-hashing bag-of-words embedding with online IDF weighting.

    Each token hashes to a bucket (with a signed hash to reduce collision
    bias); vectors are IDF-weighted so boilerplate words stop dominating.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self._doc_freq: Dict[str, int] = {}
        self._num_docs = 0

    def _bucket(self, token: str) -> Tuple[int, float]:
        h = hashlib.blake2b(token.encode(), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % self.dim
        sign = 1.0 if h[4] % 2 == 0 else -1.0
        return idx, sign

    def fit_corpus(self, texts: Sequence[str]) -> None:
        """Accumulate document frequencies (call once per indexed corpus)."""
        for text in texts:
            self._num_docs += 1
            for tok in set(tokenize(text)):
                self._doc_freq[tok] = self._doc_freq.get(tok, 0) + 1

    def _idf(self, token: str) -> float:
        df = self._doc_freq.get(token, 0)
        return math.log((1 + self._num_docs) / (1 + df)) + 1.0

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim)
        tokens = tokenize(text)
        if not tokens:
            return vec
        counts: Dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        for tok, count in counts.items():
            idx, sign = self._bucket(tok)
            vec[idx] += sign * (1 + math.log(count)) * self._idf(tok)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class VectorStore:
    """Flat cosine-similarity index. Exact search — fine into the tens of
    thousands of chunks; swap for FAISS/HNSW behind the same 3 methods beyond.
    """

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or HashingEmbedder()
        self._ids: List[int] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, chunk_id: int, text: str) -> None:
        vec = self.embedder.embed(text)
        self._ids.append(chunk_id)
        self._matrix = (
            vec[None, :]
            if self._matrix is None
            else np.vstack([self._matrix, vec[None, :]])
        )

    def add_batch(self, items: Sequence[Tuple[int, str]]) -> None:
        for chunk_id, text in items:
            self.add(chunk_id, text)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Top-k (chunk_id, cosine similarity) for the query."""
        if self._matrix is None:
            return []
        q = self.embedder.embed(query)
        sims = self._matrix @ q  # rows are unit vectors, so this is cosine
        order = np.argsort(-sims)[:top_k]
        return [(self._ids[i], float(sims[i])) for i in order]

    def remove(self, chunk_id: int) -> None:
        if chunk_id not in self._ids:
            return
        i = self._ids.index(chunk_id)
        self._ids.pop(i)
        self._matrix = np.delete(self._matrix, i, axis=0)
        if self._matrix.size == 0:
            self._matrix = None
