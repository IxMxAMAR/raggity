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
        candidates = self._hybrid(query)
        if not candidates:
            return []
        if self.cfg.rerank:
            candidates = self.reranker.rerank(query, candidates)
        survivors = [c for c in candidates if c.score >= self.cfg.relevance_floor]
        if not survivors:
            return []  # abstain signal
        survivors = dedup_chunks(survivors, self.embedder, self.cfg.dedup_cosine)
        top = survivors[: self.cfg.top_k]
        return order_lost_in_middle(top)
