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
    vec = [_chunk("c1", "relevant alpha"), _chunk("c2", "noise")]
    txt = [_chunk("c1", "relevant alpha")]
    cfg = RetrievalConfig(candidates=10, top_k=1, rerank=True, relevance_floor=0.3)
    r = Retriever(FakeEmbedder(), FakeStore(vec, txt), FakeReranker(), cfg)
    out = r.retrieve("find relevant")
    assert len(out) == 1 and out[0].chunk_id == "c1"


def test_retrieve_abstains_below_floor():
    vec = [_chunk("c2", "noise")]
    txt = [_chunk("c2", "noise")]
    cfg = RetrievalConfig(candidates=10, top_k=3, rerank=True, relevance_floor=0.3)
    r = Retriever(FakeEmbedder(), FakeStore(vec, txt), FakeReranker(), cfg)
    assert r.retrieve("q") == []  # all below floor → abstain signal


def test_retrieve_no_floor_when_rerank_off():
    # Candidates have scores below 0.3 (e.g. BM25 raw scores or low cosine).
    # With rerank=False the floor must NOT be applied — all candidates should
    # pass through dedup and top_k, not trigger abstention.
    vec = [_chunk("c1", "something useful", score=0.05),
           _chunk("c2", "other text", score=0.1)]
    txt = []
    cfg = RetrievalConfig(candidates=10, top_k=5, rerank=False,
                          relevance_floor=0.3, hybrid=False)
    r = Retriever(FakeEmbedder(), FakeStore(vec, txt), FakeReranker(), cfg)
    out = r.retrieve("q")
    assert len(out) > 0, "floor should be bypassed when rerank=False"


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
    a = _chunk("c1", "relevant alpha"); b = _chunk("c2", "relevant beta")
    # dedup_cosine > 1.0 disables dedup so both chunks survive independent of embedding similarity
    cfg = RetrievalConfig(candidates=10, top_k=2, rerank=True, relevance_floor=0.3, dedup_cosine=1.01)
    r = Retriever(FakeEmbedder(), FakeStore([a, b], [a, b]), FakeReranker(), cfg)
    out = r.retrieve_multi(["q1", "q2"], rerank_query="find relevant")
    ids = {c.chunk_id for c in out}
    assert ids == {"c1", "c2"}


def test_retrieve_parent_document_returns_parent_text():
    from raggity.config import RetrievalConfig
    from raggity.retriever import Retriever
    a = _chunk("c1", "relevant child", score=0.0)
    a.parent_id = "p1"; a.parent_text = "RELEVANT PARENT CONTEXT"
    cfg = RetrievalConfig(candidates=10, top_k=3, rerank=True,
                          relevance_floor=0.3, parent_document=True)
    r = Retriever(FakeEmbedder(), FakeStore([a], [a]), FakeReranker(), cfg)
    out = r.retrieve("find relevant")
    assert len(out) == 1
    assert out[0].text == "RELEVANT PARENT CONTEXT"
