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
    # Reuse vectors already returned by the store (vector_search/text_search/
    # get_by_chunk_ids) — only embed chunks that arrived without one (e.g. after
    # expand_to_parents swapped in different text). Index-aligned batch embed.
    missing_idx = [i for i, c in enumerate(chunks) if c.vector is None]
    if missing_idx:
        fresh = embedder.embed_documents([chunks[i].text for i in missing_idx])
        fresh_by_idx = dict(zip(missing_idx, fresh))
    else:
        fresh_by_idx = {}
    vectors = [c.vector if c.vector is not None else fresh_by_idx[i]
               for i, c in enumerate(chunks)]
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
        out.append(replace(c, text=c.parent_text, vector=None))
    return out


class Retriever:
    def __init__(self, embedder, store, reranker, cfg) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.cfg = cfg

    def retrieve(self, query: str, *, top_k: int | None = None,
                 apply_sufficiency: bool = True) -> list[Chunk]:
        return self.retrieve_multi([query], query, top_k=top_k,
                                   apply_sufficiency=apply_sufficiency)

    def retrieve_multi(self, queries: list[str], rerank_query: str,
                       graph_chunk_ids: list[str] | None = None, *,
                       top_k: int | None = None,
                       apply_sufficiency: bool = True) -> list[Chunk]:
        """Gather candidates across all queries, fuse via RRF, rerank, then return top-k.

        Abstention is keyed on the DENSE cosine similarity (reliable: relevant ~0.6–0.8,
        off-topic ~0.43–0.47).  The cross-encoder (reranker) is used only for ORDERING;
        its absolute score does NOT govern abstention.

        *top_k* overrides ``cfg.top_k`` for this call only (shared config is never
        mutated); the candidate fetch size is widened to ``max(candidates, top_k)``
        so a caller-requested k larger than ``cfg.candidates`` still works.
        *apply_sufficiency=False* bypasses the sufficiency-floor abstention (raw
        retrieval for external orchestrators that want top-k with scores).
        """
        k = self.cfg.top_k if top_k is None else top_k
        n = max(self.cfg.candidates, k)
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
            # When rerank is off, stamp the RRF fused score onto each candidate so that
            # order_lost_in_middle sorts on a single scale (not a mix of dense-cosine
            # and BM25/synthetic values). Rerank path overwrites .score anyway.
            if not self.cfg.rerank:
                candidates = [replace(c, score=fused.get(c.chunk_id, c.score))
                              for c in candidates]
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
        if not candidates:
            return []
        if apply_sufficiency and max_dense < self.cfg.sufficiency_floor:
            return []

        # Graph-augmented retrieval: merge neighborhood chunks before rerank (opt-in).
        if getattr(self.cfg, "graph", False) and graph_chunk_ids:
            existing_ids = {c.chunk_id for c in candidates}
            extra_ids = [cid for cid in graph_chunk_ids if cid not in existing_ids]
            if extra_ids:
                extra_chunks = self.store.get_by_chunk_ids(extra_ids)
                candidates = candidates + extra_chunks

        if self.cfg.rerank:
            candidates = self.reranker.rerank(rerank_query, candidates)
            # OPTIONAL secondary rerank-score filter (off by default: relevance_floor=0.0).
            # When enabled (relevance_floor > 0), filters low rerank scores but does NOT
            # abstain if this empties the list — that's governed solely by dense signal above.
            if self.cfg.relevance_floor > 0:
                candidates = [c for c in candidates if c.score >= self.cfg.relevance_floor]

        survivors = dedup_chunks(candidates, self.embedder, self.cfg.dedup_cosine)
        top = survivors[: k]
        # Expand to parents BEFORE ordering so score-based ordering (lost-in-middle)
        # acts on the final collapsed set, not on the pre-expansion child list.
        if getattr(self.cfg, "parent_document", False):
            top = expand_to_parents(top)
        ordered = order_lost_in_middle(top)
        return ordered
