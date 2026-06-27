from __future__ import annotations

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
        scores = list(self._model.rerank(query, [c.text for c in chunks]))
        scored = [replace(c, score=float(s)) for c, s in zip(chunks, scores)]
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored


register("reranker", "fastembed", "raggity.reranker:FastEmbedReranker")
