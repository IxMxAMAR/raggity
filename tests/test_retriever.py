from raggity.models import Chunk
from raggity.config import RetrievalConfig
from raggity.retriever import Retriever, dedup_chunks, order_lost_in_middle, expand_to_parents


def _chunk(cid, text, score=0.0):
    return Chunk(text=text, source_path="a.md", title="A",
                 heading_path="A", ordinal=0, chunk_id=cid, score=score)


class FakeEmbedder:
    dim = 3

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts):
        # near-identical vectors for "dup" texts; distinct for others
        out = []
        for t in texts:
            if "dup" in t:
                out.append([1.0, 0.0, 0.0])
            else:
                out.append([0.0, 1.0, 0.0])
        return out


class FakeStore:
    def __init__(self, vec, txt):
        self._vec, self._txt = vec, txt

    def vector_search(self, qv, limit):
        return self._vec[:limit]

    def text_search(self, q, limit):
        return self._txt[:limit]


def _chunk_with_dense(cid, text, dense_score, rerank_score=0.0):
    """Create a chunk whose .score is the dense cosine score (set before reranking)."""
    return Chunk(text=text, source_path="a.md", title="A",
                 heading_path="A", ordinal=0, chunk_id=cid, score=dense_score)


class FakeReranker:
    def rerank(self, query, chunks):
        # score = 1.0 if "relevant" in text else 0.1
        out = []
        for c in chunks:
            c2 = Chunk(**{**c.__dict__, "score": 1.0 if "relevant" in c.text else 0.1})
            out.append(c2)
        out.sort(key=lambda c: c.score, reverse=True)
        return out


def test_dedup_removes_near_duplicates():
    chunks = [_chunk("a", "dup text one", 0.9), _chunk("b", "dup text two", 0.8),
              _chunk("c", "different", 0.7)]
    kept = dedup_chunks(chunks, FakeEmbedder(), threshold=0.92)
    ids = {c.chunk_id for c in kept}
    assert "a" in ids and "c" in ids and "b" not in ids  # b duped against a


def test_order_lost_in_middle_puts_best_at_ends():
    chunks = [_chunk("a", "x", 0.9), _chunk("b", "x", 0.7), _chunk("c", "x", 0.5)]
    ordered = order_lost_in_middle(chunks)
    assert ordered[0].chunk_id == "a" and ordered[-1].chunk_id == "b"


def test_retrieve_returns_topk_relevant():
    # dense scores: c1=0.75 (above sufficiency_floor=0.5), c2=0.44 (below — but max is 0.75 so no abstain)
    vec = [_chunk("c1", "relevant alpha", score=0.75), _chunk("c2", "noise", score=0.44)]
    txt = [_chunk("c1", "relevant alpha", score=0.75)]
    cfg = RetrievalConfig(candidates=10, top_k=1, rerank=True, relevance_floor=0.0)
    r = Retriever(FakeEmbedder(), FakeStore(vec, txt), FakeReranker(), cfg)
    out = r.retrieve("find relevant")
    assert len(out) == 1 and out[0].chunk_id == "c1"


def test_retrieve_abstains_below_floor():
    # ALL dense scores < sufficiency_floor (0.5) → abstain regardless of reranker output
    vec = [_chunk("c2", "noise", score=0.43)]
    txt = [_chunk("c2", "noise", score=0.43)]
    cfg = RetrievalConfig(candidates=10, top_k=3, rerank=True, relevance_floor=0.0,
                          sufficiency_floor=0.5)
    r = Retriever(FakeEmbedder(), FakeStore(vec, txt), FakeReranker(), cfg)
    assert r.retrieve("q") == []  # max_dense < sufficiency_floor → abstain


def test_retrieve_no_floor_when_rerank_off():
    # Dense scores ≥ sufficiency_floor → pass dense check; relevance_floor=0.0 (off).
    # Verifies that all candidates pass through without abstention when rerank=False.
    vec = [_chunk("c1", "something useful", score=0.65),
           _chunk("c2", "other text", score=0.60)]
    txt = []
    cfg = RetrievalConfig(candidates=10, top_k=5, rerank=False,
                          relevance_floor=0.0, sufficiency_floor=0.5, hybrid=False)
    r = Retriever(FakeEmbedder(), FakeStore(vec, txt), FakeReranker(), cfg)
    out = r.retrieve("q")
    assert len(out) > 0, "candidates with dense>=0.5 should not be abstained"


def test_expand_to_parents_dedups_and_swaps_text():
    c1 = _chunk("c1", "child one", score=0.9)
    c1.parent_id = "p1"; c1.parent_text = "PARENT ONE FULL"
    c2 = _chunk("c2", "child two", score=0.5)
    c2.parent_id = "p1"; c2.parent_text = "PARENT ONE FULL"
    c3 = _chunk("c3", "child three", score=0.7)
    c3.parent_id = "p2"; c3.parent_text = "PARENT TWO FULL"
    out = expand_to_parents([c1, c2, c3])
    assert len(out) == 2                       # p1, p2
    texts = {c.text for c in out}
    assert texts == {"PARENT ONE FULL", "PARENT TWO FULL"}
    p1 = next(c for c in out if c.parent_id == "p1")
    assert p1.score == 0.9                      # best child's score kept


def test_retrieve_multi_fuses_multiple_queries():
    from raggity.config import RetrievalConfig
    from raggity.retriever import Retriever
    # dense scores ≥ 0.5 so max_dense passes sufficiency_floor
    a = _chunk("c1", "relevant alpha", score=0.70)
    b = _chunk("c2", "relevant beta", score=0.65)
    # dedup_cosine > 1.0 disables dedup so both chunks survive independent of embedding similarity
    cfg = RetrievalConfig(candidates=10, top_k=2, rerank=True, relevance_floor=0.0,
                          sufficiency_floor=0.5, dedup_cosine=1.01)
    r = Retriever(FakeEmbedder(), FakeStore([a, b], [a, b]), FakeReranker(), cfg)
    out = r.retrieve_multi(["q1", "q2"], rerank_query="find relevant")
    ids = {c.chunk_id for c in out}
    assert ids == {"c1", "c2"}


def test_retrieve_parent_document_returns_parent_text():
    from raggity.config import RetrievalConfig
    from raggity.retriever import Retriever
    # dense score ≥ 0.5 so max_dense passes sufficiency_floor; text "relevant" so reranker scores 1.0
    a = _chunk("c1", "relevant child", score=0.72)
    a.parent_id = "p1"; a.parent_text = "RELEVANT PARENT CONTEXT"
    cfg = RetrievalConfig(candidates=10, top_k=3, rerank=True,
                          relevance_floor=0.0, sufficiency_floor=0.5, parent_document=True)
    r = Retriever(FakeEmbedder(), FakeStore([a], [a]), FakeReranker(), cfg)
    out = r.retrieve("find relevant")
    assert len(out) == 1
    assert out[0].text == "RELEVANT PARENT CONTEXT"


class FakeRerankerLowAbsolute:
    """Simulates a cross-encoder that gives a low absolute score (e.g. 0.0) even for relevant chunks."""
    def rerank(self, query, chunks):
        out = []
        for c in chunks:
            # Assign 0.0 — the score a cross-encoder gives for non-question phrasings
            c2 = Chunk(**{**c.__dict__, "score": 0.0})
            out.append(c2)
        return out


def test_high_dense_low_rerank_still_returns_results():
    """Regression guard: cross-encoder absolute score of 0.0 must NOT cause abstention.

    This is the exact bug being fixed: casual phrasings like "tell me about my GPUs"
    or "GPUs" score ~0.0 on the cross-encoder even when the correct doc is top-1.
    With the NEW defaults (relevance_floor=0.0, sufficiency_floor=0.5) and max_dense >= 0.5,
    the retriever must return results regardless of the reranker's absolute score.
    """
    # dense score = 0.70 → well above sufficiency_floor 0.5
    # Uses DEFAULT RetrievalConfig — after fix: relevance_floor=0.0 (off), sufficiency_floor=0.5
    vec = [_chunk("gpu_doc", "relevant GPU document", score=0.70)]
    txt = [_chunk("gpu_doc", "relevant GPU document", score=0.70)]
    cfg = RetrievalConfig(candidates=10, top_k=3, rerank=True)  # uses defaults
    r = Retriever(FakeEmbedder(), FakeStore(vec, txt), FakeRerankerLowAbsolute(), cfg)
    out = r.retrieve("GPUs")
    assert len(out) == 1, (
        "Should return the GPU doc; cross-encoder absolute score of 0.0 must not cause abstention"
    )
    assert out[0].chunk_id == "gpu_doc"
