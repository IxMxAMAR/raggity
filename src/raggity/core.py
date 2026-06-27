from __future__ import annotations

import asyncio
import os

from .answerer import ClaudeAgentAnswerer
from .config import RaggityConfig, load_config
from .embedder import FastEmbedEmbedder
from .indexer import IngestReport, Indexer
from .models import Answer
from .retriever import Retriever
from .store import LanceDBStore


class Raggity:
    def __init__(self, cfg: RaggityConfig | None = None) -> None:
        self.cfg = cfg or RaggityConfig()
        self.embedder = FastEmbedEmbedder(
            model_name=self.cfg.embedding.model,
            provider=self.cfg.embedding.provider,
        )
        self.store = LanceDBStore(path=self.cfg.index.path, dim=self.embedder.dim)
        self.reranker = None
        if self.cfg.retrieval.rerank:
            from .reranker import FastEmbedReranker
            self.reranker = FastEmbedReranker(model_name=self.cfg.retrieval.rerank_model)
        self.retriever = Retriever(self.embedder, self.store, self.reranker,
                                   self.cfg.retrieval)
        self.answerer = ClaudeAgentAnswerer(model=self.cfg.generation.model,
                                            auth=self.cfg.generation.auth)

    @classmethod
    def from_config(cls, path: str | None = None) -> "Raggity":
        return cls(load_config(path))

    def _manifest_path(self) -> str:
        return os.path.join(self.cfg.index.path, "manifest.json")

    def ingest(self) -> IngestReport:
        indexer = Indexer(self.embedder, self.store, self._manifest_path())
        return indexer.ingest(self.cfg.sources.include)

    def ask(self, question: str) -> Answer:
        chunks = self.retriever.retrieve(question)
        return asyncio.run(self.answerer.answer(question, chunks))

    def status(self) -> dict:
        return {
            "chunks": self.store.count(),
            "sources": len(self.store.all_source_paths()),
            "index_path": self.cfg.index.path,
            "model": self.cfg.generation.model,
        }
