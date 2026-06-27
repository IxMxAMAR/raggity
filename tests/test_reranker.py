import pytest
from raggity.models import Chunk
from raggity.reranker import FastEmbedReranker


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
