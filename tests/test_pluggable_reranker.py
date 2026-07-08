"""Part A: pluggable rerankers via the dotted-path registry."""
import pytest

from raggity.config import RaggityConfig, RetrievalConfig
from raggity.core import Raggity
from raggity.models import Chunk
from raggity.registry import resolve
from raggity.reranker import ColbertReranker, FastEmbedReranker


def _chunk(cid, text, score=0.9):
    return Chunk(text=text, source_path="a.md", title="A", heading_path="A",
                 ordinal=0, chunk_id=cid, score=score)


def test_named_backends_resolve_to_correct_classes():
    assert resolve("reranker", "cross-encoder") is FastEmbedReranker
    assert resolve("reranker", "colbert") is ColbertReranker


def test_config_accepts_dotted_path_backend():
    rc = RetrievalConfig(rerank_backend="helpers_custom_reranker:SpyReranker")
    assert rc.rerank_backend == "helpers_custom_reranker:SpyReranker"


def test_config_rejects_invalid_name_without_colon():
    with pytest.raises(ValueError):
        RetrievalConfig(rerank_backend="bogus")


def test_core_reranker_resolves_dotted_custom_class():
    import helpers_custom_reranker as h

    cfg = RaggityConfig()
    cfg.retrieval.rerank_backend = "helpers_custom_reranker:SpyReranker"
    rag = Raggity(cfg)
    rr = rag.reranker
    assert isinstance(rr, h.SpyReranker)
    # Instantiated with the configured rerank_model (custom contract).
    assert rr.model_name == cfg.retrieval.rerank_model


class _FakeEmb:
    dim = 3

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts):
        return [[float(i + 1), 0.0, 0.0] for i, _ in enumerate(texts)]


class _FakeStore:
    def __init__(self, chunks):
        self._chunks = chunks

    def vector_search(self, qv, limit):
        return self._chunks[:limit]

    def text_search(self, q, limit):
        return []


def test_retriever_uses_dotted_custom_reranker(monkeypatch):
    import helpers_custom_reranker as h

    cfg = RaggityConfig()
    cfg.retrieval.rerank_backend = "helpers_custom_reranker:SpyReranker"
    cfg.retrieval.hybrid = False
    rag = Raggity(cfg)
    rag._embedder = _FakeEmb()
    rag._store = _FakeStore([_chunk("c1", "alpha", 0.9), _chunk("c2", "beta", 0.85)])

    out = rag.retriever.retrieve("hello")
    assert isinstance(rag.reranker, h.SpyReranker)
    assert rag.reranker.calls, "custom reranker was not invoked by the retriever"
    assert rag.reranker.calls[0][0] == "hello"
    assert out
