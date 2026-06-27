from __future__ import annotations

import math

from .models import Chunk
from .store import rrf_fuse


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def dedup_chunks(chunks: list[Chunk], embedder, threshold: float) -> list[Chunk]:
    if not chunks:
        return []
    vectors = embedder.embed_documents([c.text for c in chunks])
    kept: list[Chunk] = []
    kept_vecs: list[list[float]] = []
    for c, v in zip(chunks, vectors):
        if any(_cosine(v, kv) >= threshold for kv in kept_vecs):
            continue
        kept.append(c)
        kept_vecs.append(v)
    return kept


def order_lost_in_middle(chunks: list[Chunk]) -> list[Chunk]:
    ordered = sorted(chunks, key=lambda c: c.score, reverse=True)
    head: list[Chunk] = []
    tail: list[Chunk] = []
    for i, c in enumerate(ordered):
        (head if i % 2 == 0 else tail).append(c)
    return head + list(reversed(tail))


def expand_to_parents(chunks: list[Chunk]) -> list[Chunk]:
    out: list[Chunk] = []
    best_by_parent: dict[str, Chunk] = {}
    for c in chunks:
        if not c.parent_id:
            out.append(c)
            continue
        cur = best_by_parent.get(c.parent_id)
        if cur is None or c.score > cur.score:
            best_by_parent[c.parent_id] = c
    for c in best_by_parent.values():
        from dataclasses import replace
        out.append(replace(c, text=c.parent_text))
    return out


class Retriever:
    def __init__(self, embedder, store, reranker, cfg) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.cfg = cfg

    def _hybrid(self, query: str) -> list[Chunk]:
        n = self.cfg.candidates
        qv = self.embedder.embed_query(query)
        dense = self.store.vector_search(qv, n)
        if not self.cfg.hybrid:
            return dense
        sparse = self.store.text_search(query, n)
        by_id: dict[str, Chunk] = {c.chunk_id: c for c in dense}
        for c in sparse:
            by_id.setdefault(c.chunk_id, c)
        fused = rrf_fuse(
            [[c.chunk_id for c in dense], [c.chunk_id for c in sparse]],
            k=self.cfg.rrf_k,
        )
        ranked_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)
        return [by_id[cid] for cid in ranked_ids if cid in by_id][:n]

    def retrieve(self, query: str) -> list[Chunk]:
        return self.retrieve_multi([query], query)

    def retrieve_multi(self, queries: list[str], rerank_query: str) -> list[Chunk]:
        # gather candidates per query, fuse by RRF across all
        per_query_ids: list[list[str]] = []
        by_id: dict[str, Chunk] = {}
        for q in queries:
            cands = self._hybrid(q)
            per_query_ids.append([c.chunk_id for c in cands])
            for c in cands:
                by_id.setdefault(c.chunk_id, c)
        if not by_id:
            return []
        fused = rrf_fuse(per_query_ids, k=self.cfg.rrf_k)
        ranked = sorted(fused, key=lambda cid: fused[cid], reverse=True)
        candidates = [by_id[cid] for cid in ranked if cid in by_id][: self.cfg.candidates]
        if self.cfg.rerank:
            candidates = self.reranker.rerank(rerank_query, candidates)
            # Relevance floor is calibrated to reranker (sigmoid 0–1) scores.
            # Only apply it — and the floor-based abstain — when reranking is on.
            survivors = [c for c in candidates if c.score >= self.cfg.relevance_floor]
            if not survivors:
                return []  # abstain signal
        else:
            # When rerank is off, candidate scores are heterogeneous
            # (cosine ~0–1, BM25 ~0–15), so a single floor threshold is
            # meaningless. Skip the filter; abstain only when no candidates exist.
            survivors = candidates
        survivors = dedup_chunks(survivors, self.embedder, self.cfg.dedup_cosine)
        top = survivors[: self.cfg.top_k]
        ordered = order_lost_in_middle(top)
        if getattr(self.cfg, "parent_document", False):
            ordered = expand_to_parents(ordered)
        return ordered
