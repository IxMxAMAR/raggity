"""A tiny custom reranker used by the pluggable-reranker tests via a dotted
import path (``helpers_custom_reranker:SpyReranker``).

Contract mirrors :class:`raggity.reranker.Reranker`: constructed with
``model_name=...`` and exposes ``rerank(query, chunks) -> chunks`` with scores,
sorted descending. It records every call so tests can assert it was actually
used by the retriever.
"""
from __future__ import annotations

from dataclasses import replace


class SpyReranker:
    def __init__(self, model_name: str = "unset") -> None:
        self.model_name = model_name
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query, chunks):
        self.calls.append((query, [c.chunk_id for c in chunks]))
        scored = [replace(c, score=1.0 - i * 0.01) for i, c in enumerate(chunks)]
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored
