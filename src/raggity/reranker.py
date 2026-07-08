from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import replace

from .models import Chunk
from .registry import register


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]: ...


class FastEmbedReranker(Reranker):
    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []
        logits = list(self._model.rerank(query, [c.text for c in chunks]))
        scored = [replace(c, score=1.0 / (1.0 + math.exp(-float(s)))) for c, s in zip(chunks, logits)]
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored


def _l2_normalize_rows(mat):
    """L2-normalize each row (token vector) of a 2D array. Safe on zero rows."""
    import numpy as np

    mat = np.asarray(mat, dtype=np.float64)
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return mat / norms


def _maxsim_score(q, d) -> float:
    """ColBERT MaxSim: sum over query tokens of max over doc tokens of (q . d),
    normalized by the number of query tokens so the result lands in ~[0, 1]
    (each per-token max is a cosine similarity, typically positive for related
    text since both sides are unit-normalized).
    """
    if q.shape[0] == 0 or d.shape[0] == 0:
        return 0.0
    sims = q @ d.T  # (tq, td)
    per_query_max = sims.max(axis=1)  # (tq,)
    return float(per_query_max.sum()) / q.shape[0]


class ColbertReranker(Reranker):
    """Late-interaction (ColBERT-style) reranker: storage-free, rerank-stage only.

    Uses fastembed's ``LateInteractionTextEmbedding`` to produce token-level
    multi-vector embeddings for the query and each candidate chunk, then scores
    each chunk with MaxSim. Scores are normalized to roughly [0, 1] but are NOT
    comparable to :class:`FastEmbedReranker`'s sigmoid cross-encoder scores —
    relevance_floor thresholds tuned for one backend do not transfer to the other.
    """

    def __init__(self, model_name: str = "answerdotai/answerai-colbert-small-v1") -> None:
        self._model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from fastembed.late_interaction import LateInteractionTextEmbedding

            self._model = LateInteractionTextEmbedding(self._model_name)
        return self._model

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []
        model = self._ensure_model()
        q_vecs = _l2_normalize_rows(next(iter(model.query_embed(query))))
        doc_vecs = [_l2_normalize_rows(v) for v in model.embed([c.text for c in chunks])]
        scored = [replace(c, score=_maxsim_score(q_vecs, d)) for c, d in zip(chunks, doc_vecs)]
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored


register("reranker", "fastembed", "raggity.reranker:FastEmbedReranker")
register("reranker", "colbert", "raggity.reranker:ColbertReranker")
