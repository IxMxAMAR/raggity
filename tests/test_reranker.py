import os

import numpy as np
import pytest
from raggity.models import Chunk
from raggity.reranker import ColbertReranker, FastEmbedReranker, _maxsim_score


def _chunk(cid, text):
    return Chunk(text=text, source_path="a.md", title="A",
                 heading_path="A", ordinal=0, chunk_id=cid)


@pytest.fixture(scope="module")
def rr():
    return FastEmbedReranker(model_name="Xenova/ms-marco-MiniLM-L-6-v2")


def test_rerank_orders_relevant_first(rr):
    chunks = [
        _chunk("c1", "the weather is sunny and warm today"),
        _chunk("c2", "to back up your data, copy it to the NAS nightly"),
    ]
    out = rr.rerank("how do I back up my data?", chunks)
    assert out[0].chunk_id == "c2"
    assert out[0].score >= out[1].score


def test_rerank_empty_returns_empty(rr):
    assert rr.rerank("q", []) == []


def test_rerank_scores_in_zero_one(rr):
    chunks = [
        _chunk("c1", "the weather is sunny and warm today"),
        _chunk("c2", "to back up your data, copy it to the NAS nightly"),
    ]
    out = rr.rerank("how do I back up my data?", chunks)
    for c in out:
        assert 0.0 < c.score < 1.0, f"score {c.score} not in (0, 1)"


# --- ColbertReranker: MaxSim math on toy vectors (no real model download) ---

class _FakeLateInteractionModel:
    """Stand-in for fastembed's LateInteractionTextEmbedding.

    Returns fixed, already-unit-norm 2-token query vectors and per-doc token
    vectors chosen so the expected MaxSim scores are hand-computable.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        # 2 query tokens, dim=4, orthonormal basis vectors on axes 0/1.
        self._q = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    def query_embed(self, query, **kwargs):
        yield self._q

    def embed(self, documents, **kwargs):
        docs = list(documents)
        for doc in docs:
            if doc == "doc_high":
                # Perfect match on both query token axes -> MaxSim = (1+1)/2 = 1.0
                yield np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            elif doc == "doc_mid":
                # Single token: cos(q0)=0.8, cos(q1)=0.6 -> MaxSim = (0.8+0.6)/2 = 0.7
                yield np.array([[0.8, 0.6, 0.0, 0.0]])
            elif doc == "doc_low":
                # Orthogonal to both query axes (lives in dims 2/3) -> MaxSim = 0.0
                yield np.array([[0.0, 0.0, 1.0, 0.0]])
            else:
                yield np.array([[0.1, 0.1, 0.0, 0.0]])


@pytest.fixture
def colbert(monkeypatch):
    rr = ColbertReranker(model_name="fake/colbert")

    def _fake_ensure_model():
        if rr._model is None:
            rr._model = _FakeLateInteractionModel(rr._model_name)
        return rr._model

    monkeypatch.setattr(rr, "_ensure_model", _fake_ensure_model)
    return rr


def test_maxsim_score_hand_computed():
    # q has 2 tokens: [1,0] and [0,1] (already unit norm).
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    # d has 2 tokens: [1,0] and [0,1] -> each query token's best match is 1.0.
    d = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert _maxsim_score(q, d) == pytest.approx(1.0)

    # d skewed so query token 0 best-matches 0.5, query token 1 best-matches 1.0.
    d2 = np.array([[0.5, 0.0], [0.0, 1.0]])
    assert _maxsim_score(q, d2) == pytest.approx((0.5 + 1.0) / 2)


def test_maxsim_score_empty_vectors_is_zero():
    q = np.zeros((0, 4))
    d = np.array([[1.0, 0.0, 0.0, 0.0]])
    assert _maxsim_score(q, d) == 0.0
    assert _maxsim_score(np.array([[1.0, 0.0, 0.0, 0.0]]), np.zeros((0, 4))) == 0.0


def test_colbert_rerank_orders_by_maxsim_and_normalizes(colbert):
    chunks = [
        _chunk("low", "doc_low"),
        _chunk("high", "doc_high"),
        _chunk("mid", "doc_mid"),
    ]
    out = colbert.rerank("q", chunks)
    assert [c.chunk_id for c in out] == ["high", "mid", "low"]
    for c in out:
        assert 0.0 <= c.score <= 1.0, f"score {c.score} not in [0, 1]"
    scores = {c.chunk_id: c.score for c in out}
    assert scores["high"] == pytest.approx(1.0)
    assert scores["mid"] == pytest.approx(0.7)
    assert scores["low"] == pytest.approx(0.0)


def test_colbert_rerank_empty_returns_empty(colbert):
    assert colbert.rerank("q", []) == []


def test_colbert_rerank_single_chunk(colbert):
    out = colbert.rerank("q", [_chunk("only", "doc_high")])
    assert len(out) == 1
    assert out[0].chunk_id == "only"


def test_colbert_rerank_ties_stable(colbert):
    chunks = [_chunk("a", "doc_high"), _chunk("b", "doc_high")]
    out = colbert.rerank("q", chunks)
    assert [c.chunk_id for c in out] == ["a", "b"]


@pytest.mark.skipif(
    os.environ.get("RAGGITY_COLBERT_IT") != "1",
    reason="set RAGGITY_COLBERT_IT=1 to run the real colbert model download/integration test",
)
def test_colbert_rerank_real_model_smoke():
    rr = ColbertReranker(model_name="answerdotai/answerai-colbert-small-v1")
    chunks = [
        _chunk("c1", "the weather is sunny and warm today"),
        _chunk("c2", "to back up your data, copy it to the NAS nightly"),
    ]
    out = rr.rerank("how do I back up my data?", chunks)
    assert out[0].chunk_id == "c2"
    assert out[0].score >= out[1].score
