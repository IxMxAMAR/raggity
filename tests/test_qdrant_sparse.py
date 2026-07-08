"""Part B: Qdrant learned-sparse retrieval (splade/bm42).

Uses a fake ``SparseTextEmbedding`` (deterministic indices/values) so no real
0.5GB SPLADE model is downloaded. One gated real-model test uses the small
bm42 (~0.09GB) behind ``RAGGITY_SPARSE_IT=1``.
"""
import logging
import os
from dataclasses import dataclass

import pytest

from raggity.config import RaggityConfig
from raggity.core import Raggity
from raggity.embedder import FastEmbedEmbedder
from raggity.models import Chunk
from raggity.qdrant_store import QdrantStore
from raggity.store import LanceDBStore


@dataclass
class _SE:
    indices: list
    values: list


class _FakeSparse:
    """Deterministic fake: token = char-code bucket; value 1.0 per token."""

    def __init__(self):
        self.doc_calls = 0
        self.query_calls = 0

    @staticmethod
    def _tokens(text):
        idx = sorted({ord(ch) % 97 for ch in text.lower() if ch.isalnum()})
        return idx or [0]

    def embed(self, texts, **kw):
        for t in texts:
            self.doc_calls += 1
            idx = self._tokens(t)
            yield _SE(indices=idx, values=[1.0] * len(idx))

    def query_embed(self, query, **kw):
        self.query_calls += 1
        idx = self._tokens(query)
        yield _SE(indices=idx, values=[1.0] * len(idx))


@pytest.fixture(scope="module")
def emb():
    return FastEmbedEmbedder()


def _chunk(cid, text, src="a.md", ordinal=0):
    return Chunk(text=text, source_path=src, title="A", heading_path="A",
                 ordinal=ordinal, chunk_id=cid)


def _sparse_store(dim, choice="splade"):
    return QdrantStore(location=":memory:", dim=dim, collection="t",
                       sparse=choice, sparse_embedder=_FakeSparse())


def test_default_store_has_no_sparse_config(emb):
    s = QdrantStore(location=":memory:", dim=emb.dim, collection="t")
    info = s._client.get_collection("t")
    assert not info.config.params.sparse_vectors


def test_sparse_collection_created_with_sparse_config(emb):
    s = _sparse_store(emb.dim)
    info = s._client.get_collection("t")
    assert "sparse" in info.config.params.sparse_vectors


def test_upsert_attaches_sparse_vectors(emb):
    s = _sparse_store(emb.dim)
    s.upsert([_chunk("c1", "backups run nightly to the NAS")], emb)
    pid = s._pid("c1")
    pt = s._client.retrieve("t", ids=[pid], with_vectors=True)[0]
    assert isinstance(pt.vector, dict)
    assert "sparse" in pt.vector
    assert len(pt.vector["sparse"].indices) > 0
    # Dense component still present and correctly sized.
    assert len(pt.vector[""]) == emb.dim


def test_text_search_returns_sparse_ranked_results(emb):
    s = _sparse_store(emb.dim)
    s.upsert([_chunk("c1", "rotate the API key yearly"),
              _chunk("c2", "zzz", ordinal=1)], emb)
    res = s.text_search("rotate the API key", limit=5)
    assert res, "sparse text_search returned nothing"
    assert res[0].chunk_id == "c1"
    # Chunk vector is the dense component (dict unpacked), not the raw dict.
    assert res[0].vector is not None and len(res[0].vector) == emb.dim
    assert s._sparse_model.query_calls >= 1


def test_bm25_text_search_unchanged(emb):
    """bm25 default keeps the MatchText scroll path (no sparse model touched)."""
    s = QdrantStore(location=":memory:", dim=emb.dim, collection="t")
    s.upsert([_chunk("c1", "rotated the API key on 2026"),
              _chunk("c2", "unrelated content", ordinal=1)], emb)
    res = s.text_search("API key", limit=5)
    assert any(c.chunk_id == "c1" for c in res)
    assert res[0].vector is not None and len(res[0].vector) == emb.dim


def test_lancedb_sparse_falls_back_to_bm25_with_warning(tmp_path, caplog):
    cfg = RaggityConfig()
    cfg.index.backend = "lancedb"
    cfg.index.path = str(tmp_path / "idx")
    cfg.retrieval.sparse = "splade"
    rag = Raggity(cfg)
    with caplog.at_level(logging.WARNING):
        store = rag.store
    assert isinstance(store, LanceDBStore)
    assert any("sparse" in r.message.lower() and "bm25" in r.message.lower()
               for r in caplog.records)


def test_fingerprint_toggles_with_sparse_choice():
    cfg = RaggityConfig()
    cfg.index.backend = "qdrant"
    rag = Raggity(cfg)
    base = rag._fingerprint_with_dim(384)

    cfg2 = RaggityConfig()
    cfg2.index.backend = "qdrant"
    cfg2.retrieval.sparse = "splade"
    rag2 = Raggity(cfg2)
    assert rag2._fingerprint_with_dim(384) != base
    assert "sparse=splade" in rag2._fingerprint_with_dim(384)


def test_fingerprint_default_unchanged_for_lancedb_sparse():
    """Sparse marker is qdrant-only; lancedb fingerprint stays byte-identical."""
    cfg = RaggityConfig()  # lancedb default
    rag = Raggity(cfg)
    base = rag._fingerprint_with_dim(384)

    cfg2 = RaggityConfig()
    cfg2.retrieval.sparse = "bm42"
    rag2 = Raggity(cfg2)
    assert rag2._fingerprint_with_dim(384) == base


@pytest.mark.skipif(
    os.environ.get("RAGGITY_SPARSE_IT") != "1",
    reason="set RAGGITY_SPARSE_IT=1 to run the real bm42 sparse model download/integration test",
)
def test_real_bm42_sparse_smoke(emb):
    s = QdrantStore(location=":memory:", dim=emb.dim, collection="t", sparse="bm42")
    s.upsert([_chunk("c1", "to back up your data copy it to the NAS nightly"),
              _chunk("c2", "the weather is sunny and warm today", ordinal=1)], emb)
    res = s.text_search("how do I back up my data", limit=5)
    assert res and res[0].chunk_id == "c1"
