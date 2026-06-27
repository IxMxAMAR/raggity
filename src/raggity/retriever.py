from __future__ import annotations

import math
from dataclasses import replace

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
        out.append(replace(c, text=c.parent_text))
    return out


class Retriever:
    def __init__(self, embedder, store, reranker, cfg) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.cfg = cfg

    def retrieve(self, query: str) -> list[Chunk]:
        return self.retrieve_multi([query], query)

    def retrieve_multi(self, queries: list[str], rerank_query: str) -> list[Chunk]:
        """Gather candidates across all queries, fuse via RRF, rerank, then return top-k.

        Abstention is keyed on the DENSE cosine similarity (reliable: relevant ~0.6–0.8,
        off-topic ~0.43–0.47).  The cross-encoder (reranker) is used only for ORDERING;
        its absolute score does NOT govern abstention.
        """
        n = self.cfg.candidates
        per_query_dense_ids: list[list[str]] = []
        per_query_sparse_ids: list[list[str]] = []
        by_id: dict[str, Chunk] = {}
        max_dense: float = 0.0

        for q in queries:
            qv = self.embedder.embed_query(q)
            dense = self.store.vector_search(qv, n)
            # Track max dense score across all queries for abstention decision
            if dense:
                max_dense = max(max_dense, max(c.score for c in dense))
            dense_ids = [c.chunk_id for c in dense]
            per_query_dense_ids.append(dense_ids)
            for c in dense:
                by_id.setdefault(c.chunk_id, c)

            if self.cfg.hybrid:
                sparse = self.store.text_search(q, n)
                sparse_ids = [c.chunk_id for c in sparse]
                per_query_sparse_ids.append(sparse_ids)
                for c in sparse:
                    by_id.setdefault(c.chunk_id, c)
            else:
                per_query_sparse_ids.append([])

        # Build ranked candidate list
        if self.cfg.hybrid:
            # Fuse dense + sparse lists from all queries
            all_id_lists = per_query_dense_ids + per_query_sparse_ids
            fused = rrf_fuse(all_id_lists, k=self.cfg.rrf_k)
            ranked_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)
            candidates = [by_id[cid] for cid in ranked_ids if cid in by_id][:n]
        else:
            # Dense-only: preserve order from first query's dense results
            all_ids: list[str] = []
            seen: set[str] = set()
            for id_list in per_query_dense_ids:
                for cid in id_list:
                    if cid not in seen:
                        all_ids.append(cid)
                        seen.add(cid)
            candidates = [by_id[cid] for cid in all_ids if cid in by_id][:n]

        # ABSTAIN: dense cosine is the reliable relevance signal.
        # If max_dense < sufficiency_floor, the top retrieved doc is not relevant enough.
        if not candidates or max_dense < self.cfg.sufficiency_floor:
            return []

        if self.cfg.rerank:
            candidates = self.reranker.rerank(rerank_query, candidates)
            # OPTIONAL secondary rerank-score filter (off by default: relevance_floor=0.0).
            # When enabled (relevance_floor > 0), filters low rerank scores but does NOT
            # abstain if this empties the list — that's governed solely by dense signal above.
            if self.cfg.relevance_floor > 0:
                candidates = [c for c in candidates if c.score >= self.cfg.relevance_floor]

        survivors = dedup_chunks(candidates, self.embedder, self.cfg.dedup_cosine)
        top = survivors[: self.cfg.top_k]
        ordered = order_lost_in_middle(top)
        if getattr(self.cfg, "parent_document", False):
            ordered = expand_to_parents(ordered)
        return ordered
