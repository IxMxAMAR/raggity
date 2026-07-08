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


class SpyEmbedder(FakeEmbedder):
    """Records embed_documents calls so tests can assert exactly what was embedded."""
    def __init__(self):
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return super().embed_documents(texts)


def _chunk_vec(cid, text, vector, score=0.0):
    return Chunk(text=text, source_path="a.md", title="A", heading_path="A",
                 ordinal=0, chunk_id=cid, score=score, vector=vector)


def test_dedup_with_all_stored_vectors_never_calls_embed_documents():
    """When every chunk already carries a stored vector, dedup must reuse it and
    never call embed_documents (the ~2.8s/query cost being removed)."""
    spy = SpyEmbedder()
    chunks = [
        _chunk_vec("a", "dup text one", [1.0, 0.0, 0.0], 0.9),
        _chunk_vec("b", "dup text two", [1.0, 0.0, 0.0], 0.8),
        _chunk_vec("c", "different", [0.0, 1.0, 0.0], 0.7),
    ]
    kept = dedup_chunks(chunks, spy, threshold=0.92)
    ids = {c.chunk_id for c in kept}
    assert ids == {"a", "c"}  # b duped against a — identical to the embedding path
    assert spy.calls == []


def test_dedup_embeds_only_chunks_missing_vector():
    """Mixed set: exactly one embed_documents call, containing exactly the missing
    chunk's text, with index alignment preserved."""
    spy = SpyEmbedder()
    chunks = [
        _chunk_vec("a", "dup text one", [1.0, 0.0, 0.0], 0.9),
        _chunk("b", "dup text two", 0.8),  # vector=None (default) — not embedded yet
        _chunk_vec("c", "different", [0.0, 1.0, 0.0], 0.7),
    ]
    kept = dedup_chunks(chunks, spy, threshold=0.92)
    assert spy.calls == [["dup text two"]]
    ids = {c.chunk_id for c in kept}
    assert ids == {"a", "c"}  # b's freshly-embedded vector dupes against a's stored one


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


def test_expand_to_parents_nulls_vector():
    """Parent text no longer matches the child's stored vector — must be cleared
    so downstream dedup re-embeds the (now different) parent text."""
    c1 = _chunk_vec("c1", "child one", [1.0, 0.0, 0.0], score=0.9)
    c1.parent_id = "p1"; c1.parent_text = "PARENT ONE FULL"
    out = expand_to_parents([c1])
    assert len(out) == 1
    assert out[0].vector is None


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


class FakeStoreWithGraphIds:
    """FakeStore that also implements get_by_chunk_ids."""
    def __init__(self, vec, txt, graph_chunks=None):
        self._vec, self._txt = vec, txt
        self._graph_chunks = {c.chunk_id: c for c in (graph_chunks or [])}

    def vector_search(self, qv, limit):
        return self._vec[:limit]

    def text_search(self, q, limit):
        return self._txt[:limit]

    def get_by_chunk_ids(self, ids):
        return [self._graph_chunks[cid] for cid in ids if cid in self._graph_chunks]


def test_retrieve_merges_graph_neighborhood():
    """When cfg.graph=True and graph_chunk_ids provided, extra chunk is merged before rerank."""
    # hybrid hit: c1 (score 0.7); graph neighborhood: c2 (not in hybrid hits)
    c1 = _chunk("c1", "relevant alpha", score=0.7)
    c2 = _chunk("c2", "graph extra chunk", score=0.0)

    store = FakeStoreWithGraphIds([c1], [c1], graph_chunks=[c2])
    cfg = RetrievalConfig(candidates=10, top_k=5, rerank=True, relevance_floor=0.0,
                          sufficiency_floor=0.5, dedup_cosine=1.01, graph=True, graph_hops=1)
    r = Retriever(FakeEmbedder(), store, FakeReranker(), cfg)

    out = r.retrieve_multi(["q"], "q", graph_chunk_ids=["c2"])
    ids = {c.chunk_id for c in out}
    assert "c1" in ids, "hybrid hit must be present"
    assert "c2" in ids, "graph neighborhood chunk must be merged in"


def test_retrieve_graph_off_ignores_graph_chunk_ids():
    """When cfg.graph=False, graph_chunk_ids are NOT merged (default behavior unchanged)."""
    c1 = _chunk("c1", "relevant alpha", score=0.7)
    c2 = _chunk("c2", "graph extra chunk", score=0.0)

    store = FakeStoreWithGraphIds([c1], [c1], graph_chunks=[c2])
    # graph=False (default)
    cfg = RetrievalConfig(candidates=10, top_k=5, rerank=True, relevance_floor=0.0,
                          sufficiency_floor=0.5, dedup_cosine=1.01)
    r = Retriever(FakeEmbedder(), store, FakeReranker(), cfg)

    out = r.retrieve_multi(["q"], "q", graph_chunk_ids=["c2"])
    ids = {c.chunk_id for c in out}
    assert "c1" in ids
    assert "c2" not in ids, "graph chunk must NOT be merged when cfg.graph=False"


def test_retrieve_graph_deduplicates_overlap():
    """If graph neighborhood chunk already in hybrid hits, no duplicate in result."""
    c1 = _chunk("c1", "relevant alpha", score=0.7)

    store = FakeStoreWithGraphIds([c1], [c1], graph_chunks=[c1])
    cfg = RetrievalConfig(candidates=10, top_k=5, rerank=True, relevance_floor=0.0,
                          sufficiency_floor=0.5, dedup_cosine=1.01, graph=True, graph_hops=1)
    r = Retriever(FakeEmbedder(), store, FakeReranker(), cfg)

    out = r.retrieve_multi(["q"], "q", graph_chunk_ids=["c1"])
    chunk_ids = [c.chunk_id for c in out]
    assert chunk_ids.count("c1") == 1, "chunk_id must not appear twice"


class FakeRerankerLowAbsolute:
    """Simulates a cross-encoder that gives a low absolute score (e.g. 0.0) even for relevant chunks."""
    def rerank(self, query, chunks):
        out = []
        for c in chunks:
            # Assign 0.0 — the score a cross-encoder gives for non-question phrasings
            c2 = Chunk(**{**c.__dict__, "score": 0.0})
            out.append(c2)
        return out


def test_parent_document_respects_score_ordering():
    """parent_document=True + mix of parented and non-parented chunks:
    best-scored chunk (even when parented) must appear at head/tail.

    The bug: old code ran order_lost_in_middle(top) then expand_to_parents(ordered).
    expand_to_parents appends ALL parented chunks AFTER non-parented chunks, so
    even the highest-scored chunk (if parented) ends up after lower-scored orphans.

    Fix: expand_to_parents first, then order_lost_in_middle on the expanded result."""
    from raggity.retriever import order_lost_in_middle, expand_to_parents

    # Scenario: c_p1 (parented, score=0.9 best) + c_orphan (non-parented, score=0.6)
    # Bug: old order(top) → expand puts c_orphan at [0] (non-parented goes first), p1 after
    # Fix: expand(top) first collapses parents in-place, then order → p1 at [0]

    c_p1 = _chunk("c_p1", "parent one child", score=0.9)
    c_p1.parent_id = "p1"; c_p1.parent_text = "PARENT ONE FULL TEXT"
    c_orphan = _chunk("c_orphan", "standalone chunk", score=0.6)
    # c_orphan has no parent_id (default None from Chunk)

    input_chunks = [c_p1, c_orphan]

    # OLD code: order then expand
    old_ordered = order_lost_in_middle(input_chunks)  # [c_p1(0.9), c_orphan(0.6)]
    old_result = expand_to_parents(old_ordered)
    # OLD: orphan goes to 'out' first (position 0), then p1_expanded appended after
    assert old_result[0].chunk_id == "c_orphan", (
        "Confirming old bug: non-parented orphan (score=0.6) ends up at head, "
        "not the best-scored parented chunk p1 (0.9)"
    )

    # NEW code: expand then order → p1_exp (score=0.9) at head
    new_expanded = expand_to_parents(input_chunks)   # [c_orphan, p1_exp(score=0.9)]
    new_result = order_lost_in_middle(new_expanded)  # p1_exp at head
    assert new_result[0].parent_id == "p1", (
        f"New code: best-scored parented chunk p1 must be at head, got {new_result[0]}"
    )


def test_retriever_parent_document_best_at_head():
    """Integration: Retriever with parent_document=True, 3 chunks: best must be at head.

    Bug: old code ran order_lost_in_middle then expand_to_parents.
    With 3 chunks: [c_p1(0.9), c_p2(0.7), c_orphan(0.6)] → lost-in-middle →
    [c_p1, c_orphan, c_p2] → expand_to_parents → [c_orphan, p1_exp, p2_exp]
    → c_orphan at head (wrong: lower score than p1).

    Fix: expand first → [c_orphan, p1_exp, p2_exp] (all present, p1 score=0.9) →
    lost-in-middle → [p1_exp, c_orphan, p2_exp] → p1 at head (correct)."""
    from raggity.config import RetrievalConfig

    c_p1 = _chunk("c_p1", "relevant parent child", score=0.9)
    c_p1.parent_id = "p1"; c_p1.parent_text = "PARENT ONE FULL TEXT"
    c_p2 = _chunk("c_p2", "other parent child", score=0.7)
    c_p2.parent_id = "p2"; c_p2.parent_text = "PARENT TWO FULL TEXT"
    c_orphan = _chunk("c_orphan", "standalone chunk", score=0.6)

    class _IdentityReranker:
        def rerank(self, query, cands):
            return sorted(cands, key=lambda c: c.score, reverse=True)

    store = FakeStore([c_p1, c_p2, c_orphan], [])
    cfg = RetrievalConfig(candidates=10, top_k=3, rerank=True, relevance_floor=0.0,
                          sufficiency_floor=0.5, dedup_cosine=1.01, parent_document=True)
    r = Retriever(FakeEmbedder(), store, _IdentityReranker(), cfg)
    out = r.retrieve("find relevant")

    assert len(out) == 3
    # After fix: p1_exp (score=0.9) at head (best score)
    assert out[0].chunk_id == "c_p1", (
        f"Best-scored chunk c_p1 (0.9) must be at head after fix, "
        f"got order: {[c.chunk_id for c in out]}"
    )


def test_hybrid_no_rerank_order_by_rrf_score():
    """hybrid=True + rerank=False: final order must reflect RRF fused score,
    not the raw dense-cosine .score values.

    Bug: old code kept original .score on candidates; order_lost_in_middle sorts on
    .score which may mix scales → chunk with high dense cosine but low RRF ends at head.
    Fix: assign RRF fused score to .score before ordering."""
    from raggity.config import RetrievalConfig
    from raggity.store import rrf_fuse

    # Scenario designed to expose the bug:
    #   c_dense_only: raw score=0.85 (high dense), NOT in sparse → low RRF
    #   c_both:       raw score=0.6 (lower dense), rank=1 dense + rank=1 sparse → high RRF
    # Old code: order by raw .score → c_dense_only(0.85) at head (wrong)
    # Fixed code: order by RRF score → c_both at head (correct)

    c_dense_only = _chunk("c_dense_only", "dense only chunk", score=0.85)
    c_both = _chunk("c_both", "hybrid chunk", score=0.60)
    c_low = _chunk("c_low", "low everything", score=0.55)

    class _HybridStore:
        def vector_search(self, qv, limit):
            # Dense ranks: c_dense_only=1(score=0.85), c_both=2(0.60), c_low=3(0.55)
            return [c_dense_only, c_both, c_low][:limit]
        def text_search(self, q, limit):
            # Sparse: c_both=1, no c_dense_only
            return [c_both, c_low][:limit]

    cfg = RetrievalConfig(candidates=10, top_k=3, rerank=False, relevance_floor=0.0,
                          sufficiency_floor=0.5, dedup_cosine=1.01, hybrid=True)
    r = Retriever(FakeEmbedder(), _HybridStore(), None, cfg)
    out = r.retrieve("test query")

    assert len(out) == 3

    # Compute expected RRF fused scores to know what "correct" order is
    dense_ids = ["c_dense_only", "c_both", "c_low"]
    sparse_ids = ["c_both", "c_low"]
    fused = rrf_fuse([dense_ids, sparse_ids], k=cfg.rrf_k)
    # c_both appears in both lists → higher RRF than c_dense_only (dense-only)
    assert fused["c_both"] > fused["c_dense_only"], (
        "Test setup: c_both must have higher RRF than c_dense_only"
    )

    # With fix: .score on output reflects RRF → c_both (higher RRF) at head or tail
    # Specifically c_dense_only must NOT be at head (it has high raw score but low RRF)
    score_by_id = {c.chunk_id: c.score for c in out}
    # After fix, scores should be RRF scores (< 1.0, not raw cosine)
    assert score_by_id["c_both"] > score_by_id["c_dense_only"], (
        f"After fix: c_both RRF score must exceed c_dense_only RRF score; "
        f"got c_both={score_by_id['c_both']:.4f}, c_dense_only={score_by_id['c_dense_only']:.4f}"
    )
    # Head must be c_both or c_low (by RRF), NOT c_dense_only
    assert out[0].chunk_id != "c_dense_only", (
        f"c_dense_only (high raw score, low RRF) must not be at head; "
        f"got order: {[c.chunk_id for c in out]}"
    )


# ---------------------------------------------------------------------------
# RED: v0.11.0 T2 — top_k override + sufficiency bypass (server /retrieve)
# ---------------------------------------------------------------------------

def _five_chunks():
    return [_chunk(f"c{i}", f"distinct text number {i}", score=0.7 - i * 0.01)
            for i in range(5)]


def test_retrieve_top_k_override_limits_results():
    """retrieve(top_k=2) returns at most 2 chunks and does NOT mutate cfg.top_k."""
    chunks = _five_chunks()
    cfg = RetrievalConfig(candidates=10, top_k=5, rerank=False, hybrid=False,
                          sufficiency_floor=0.5, dedup_cosine=1.01)
    r = Retriever(FakeEmbedder(), FakeStore(chunks, []), None, cfg)
    out = r.retrieve("q", top_k=2)
    assert len(out) == 2
    assert cfg.top_k == 5  # shared config untouched


def test_retrieve_top_k_default_uses_cfg():
    """retrieve() without top_k slices at cfg.top_k as before."""
    chunks = _five_chunks()
    cfg = RetrievalConfig(candidates=10, top_k=3, rerank=False, hybrid=False,
                          sufficiency_floor=0.5, dedup_cosine=1.01)
    r = Retriever(FakeEmbedder(), FakeStore(chunks, []), None, cfg)
    assert len(r.retrieve("q")) == 3


def test_retrieve_top_k_exceeding_candidates_widens_fetch():
    """top_k > cfg.candidates: fetch size becomes max(candidates, top_k) so k wins."""
    chunks = _five_chunks()
    cfg = RetrievalConfig(candidates=2, top_k=1, rerank=False, hybrid=False,
                          sufficiency_floor=0.5, dedup_cosine=1.01)
    r = Retriever(FakeEmbedder(), FakeStore(chunks, []), None, cfg)
    out = r.retrieve("q", top_k=4)
    assert len(out) == 4


def test_retrieve_apply_sufficiency_false_bypasses_floor():
    """apply_sufficiency=False returns candidates even below sufficiency_floor."""
    low = [_chunk("c1", "off topic", score=0.43)]
    cfg = RetrievalConfig(candidates=10, top_k=3, rerank=False, hybrid=False,
                          sufficiency_floor=0.5, dedup_cosine=1.01)
    r = Retriever(FakeEmbedder(), FakeStore(low, []), None, cfg)
    assert r.retrieve("q") == []                       # default still abstains
    out = r.retrieve("q", apply_sufficiency=False)     # bypass returns the chunk
    assert [c.chunk_id for c in out] == ["c1"]


def test_retrieve_apply_sufficiency_false_empty_store_still_empty():
    """Bypassing the floor never invents results: empty store -> []."""
    cfg = RetrievalConfig(candidates=10, top_k=3, rerank=False, hybrid=False,
                          dedup_cosine=1.01)
    r = Retriever(FakeEmbedder(), FakeStore([], []), None, cfg)
    assert r.retrieve("q", apply_sufficiency=False) == []


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
