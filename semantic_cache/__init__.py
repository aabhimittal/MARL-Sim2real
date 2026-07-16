"""Graph-based semantic cache for ultra-long contexts.

A vector-graph hybrid: embeddings find semantically similar entry points,
the graph expands them into coherent "context clusters" (things that belong
together even when they aren't lexically similar), and the cache manager
keeps the working set bounded with score-based eviction.
"""

from semantic_cache.chunker import Chunk, chunk_text
from semantic_cache.vector_store import VectorStore, HashingEmbedder
from semantic_cache.graph_store import GraphStore
from semantic_cache.hybrid_retriever import HybridRetriever, ContextCluster
from semantic_cache.cache_manager import SemanticCache

__all__ = [
    "Chunk",
    "chunk_text",
    "VectorStore",
    "HashingEmbedder",
    "GraphStore",
    "HybridRetriever",
    "ContextCluster",
    "SemanticCache",
]
