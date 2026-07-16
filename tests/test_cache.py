"""Tests for the semantic_cache package (chunker, vector_store, graph_store,
hybrid_retriever, cache_manager)."""

from __future__ import annotations

import pytest

from semantic_cache.cache_manager import SemanticCache
from semantic_cache.chunker import Chunk, chunk_text
from semantic_cache.graph_store import GraphStore
from semantic_cache.vector_store import HashingEmbedder, VectorStore

# ------------------------------------------------------------------------ chunker


def test_chunk_text_overlap_shares_words_between_neighbors():
    words = [f"word{i}" for i in range(300)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        tail = chunks[i].tokens[-10:]
        head = chunks[i + 1].tokens[:10]
        assert tail == head


def test_chunk_text_assigns_sequential_positions_and_doc_id():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, doc_id="mydoc", chunk_size=40, overlap=5, start_id=100)
    for i, c in enumerate(chunks):
        assert c.doc_id == "mydoc"
        assert c.position == i
        assert c.chunk_id == 100 + i


def test_chunk_text_rejects_overlap_ge_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=10, overlap=10)


def test_chunk_text_short_text_single_chunk():
    chunks = chunk_text("just a few words here", chunk_size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "just a few words here"


# --------------------------------------------------------------------- vector_store


def test_vector_store_search_ranks_exact_text_match_first():
    embedder = HashingEmbedder(dim=256)
    texts = {
        1: "the quick brown fox jumps over the lazy dog",
        2: "stock markets rose sharply today amid economic optimism",
        3: "deep learning models require large amounts of training data",
    }
    embedder.fit_corpus(list(texts.values()))
    store = VectorStore(embedder)
    for cid, t in texts.items():
        store.add(cid, t)

    results = store.search(texts[3], top_k=3)
    assert results[0][0] == 3
    assert results[0][1] == pytest.approx(1.0, abs=1e-6)
    # The exact match strictly beats the unrelated documents.
    assert results[0][1] > results[1][1]


def test_vector_store_remove():
    embedder = HashingEmbedder(dim=128)
    store = VectorStore(embedder)
    store.add(1, "alpha beta gamma")
    store.add(2, "delta epsilon zeta")
    assert len(store) == 2
    store.remove(1)
    assert len(store) == 1
    results = store.search("alpha beta gamma", top_k=5)
    assert all(cid != 1 for cid, _ in results)


def test_vector_store_empty_search_returns_empty():
    store = VectorStore()
    assert store.search("anything") == []


# ---------------------------------------------------------------------- graph_store


def _chain_chunks(n=3, doc_id="d"):
    return [Chunk(chunk_id=i, text=f"text {i}", doc_id=doc_id, position=i) for i in range(n)]


def test_spread_activation_reaches_two_hop_neighbors():
    g = GraphStore()
    for c in _chain_chunks(3):
        g.add_chunk(c)  # sequential edges: 0-1, 1-2

    energy = g.spread_activation({0: 1.0}, hops=2, decay=0.5)
    assert energy.get(1, 0.0) > 0.0
    assert energy.get(2, 0.0) > 0.0
    # The direct 1-hop neighbor should end up with more energy than the 2-hop one.
    assert energy[1] > energy[2]


def test_spread_activation_zero_hops_returns_only_seeds():
    g = GraphStore()
    for c in _chain_chunks(3):
        g.add_chunk(c)
    energy = g.spread_activation({0: 1.0}, hops=0)
    assert energy == {0: 1.0}


def test_remove_chunk_cleans_adjacency():
    g = GraphStore()
    for c in _chain_chunks(3):
        g.add_chunk(c)
    assert 1 in g.neighbors(0)
    g.remove_chunk(1)
    assert 1 not in g.nodes
    assert len(g) == 2
    assert 1 not in g.neighbors(0)
    assert 1 not in g.neighbors(2)


def test_entity_edges_link_chunks_sharing_entities():
    g = GraphStore()
    c0 = Chunk(chunk_id=0, text="", doc_id="a", position=0, entities={"pybullet"})
    c1 = Chunk(chunk_id=1, text="", doc_id="b", position=5, entities={"pybullet"})
    g.add_chunk(c0)
    g.add_chunk(c1)
    assert 1 in g.neighbors(0)


def test_connected_component_respects_allowed_set():
    g = GraphStore()
    for c in _chain_chunks(4):
        g.add_chunk(c)  # chain 0-1-2-3
    comp = g.connected_component(0, allowed={0, 1, 2, 3})
    assert comp == {0, 1, 2, 3}
    comp_restricted = g.connected_component(0, allowed={0, 1})
    assert comp_restricted == {0, 1}


# ------------------------------------------------------------------- SemanticCache


CATS_DOC = " ".join(
    [
        "Cats are small domesticated carnivorous mammals known for their whiskers and agility.",
        "A house cat spends much of its day grooming its soft fur and napping in sunny spots.",
        "Many cat owners describe their pet as an independent but affectionate companion animal.",
        "The whiskers of a cat are highly sensitive and help it navigate in low light conditions.",
        "Kittens play by pouncing and chasing to develop the hunting reflexes of an adult cat.",
    ]
    * 3
)

CARS_DOC = " ".join(
    [
        "Modern automobiles rely on an internal combustion engine or an electric motor for power.",
        "A typical car has four wheels, a chassis, a transmission, and a braking system.",
        "Electric vehicles store energy in a large battery pack instead of a fuel tank.",
        "Automotive engineers test crash safety, aerodynamics, and fuel efficiency before release.",
        "Sedans, hatchbacks, and pickup trucks are common body styles for passenger vehicles.",
    ]
    * 3
)


def test_cache_ingest_and_query_returns_relevant_cluster():
    cache = SemanticCache(chunk_size=40, overlap=10, max_chunks=1000)
    cache.ingest(CATS_DOC, doc_id="cats")
    cache.ingest(CARS_DOC, doc_id="cars")

    clusters = cache.query("Tell me about cats and their whiskers")
    assert clusters
    all_text = " ".join(c.text.lower() for cluster in clusters for c in cluster.chunks)
    assert "cat" in all_text


def test_cache_cluster_seed_ids_are_subset_of_cluster_chunks():
    cache = SemanticCache(chunk_size=40, overlap=10, max_chunks=1000)
    cache.ingest(CATS_DOC, doc_id="cats")
    cache.ingest(CARS_DOC, doc_id="cars")
    clusters = cache.query("cats whiskers grooming")
    for cluster in clusters:
        member_ids = {c.chunk_id for c in cluster.chunks}
        assert set(cluster.seed_ids) <= member_ids


def test_cache_render_non_empty_and_respects_budget():
    cache = SemanticCache(chunk_size=40, overlap=10, max_chunks=1000)
    cache.ingest(CATS_DOC, doc_id="cats")
    cache.ingest(CARS_DOC, doc_id="cars")
    clusters = cache.query("cats whiskers")
    assert clusters

    big_budget_text = cache.render(clusters, budget_chars=100_000)
    assert len(big_budget_text) > 0

    tiny_budget_text = cache.render(clusters, budget_chars=50)
    assert len(tiny_budget_text) == 50  # first (best) cluster gets truncated to the budget


def test_cache_render_empty_clusters_is_empty_string():
    cache = SemanticCache()
    assert cache.render([]) == ""


def test_cache_eviction_keeps_len_under_max_chunks():
    cache = SemanticCache(max_chunks=5, chunk_size=20, overlap=5)
    big_text = " ".join(f"word{i}" for i in range(600))
    cache.ingest(big_text, doc_id="big")
    assert len(cache.graph) <= 5
    assert cache.stats.chunks == len(cache.graph)
    assert cache.stats.evictions > 0


def test_cache_stats_track_docs_and_queries():
    cache = SemanticCache(chunk_size=40, overlap=10)
    cache.ingest(CATS_DOC, doc_id="cats")
    cache.ingest(CARS_DOC, doc_id="cars")
    cache.query("cats")
    cache.query("cars")
    assert cache.stats.ingested_docs == 2
    assert cache.stats.queries == 2
    assert cache.stats.chunks == len(cache.graph)
