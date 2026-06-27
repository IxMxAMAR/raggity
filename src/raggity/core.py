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

    def _fingerprint(self) -> str:
        rc = self.cfg.retrieval
        return (f"{self.cfg.embedding.model}|{self.embedder.dim}|"
                f"pd={rc.parent_document}|pt={rc.parent_target_tokens}|ct={rc.child_target_tokens}")

    def ingest(self) -> IngestReport:
        chunk_kwargs = {"parent_document": self.cfg.retrieval.parent_document,
                        "parent_target_tokens": self.cfg.retrieval.parent_target_tokens,
                        "child_target_tokens": self.cfg.retrieval.child_target_tokens}
        indexer = Indexer(self.embedder, self.store, self._manifest_path(),
                          fingerprint=self._fingerprint(), chunk_kwargs=chunk_kwargs)
        return indexer.ingest(self.cfg.sources.include)

    def ask(self, question: str, expand: bool | None = None) -> Answer:
        return asyncio.run(self.aask(question, expand))

    async def aask(self, question: str, expand: bool | None = None) -> Answer:
        use_expand = self.cfg.retrieval.expand if expand is None else expand
        if use_expand:
            from .query_transform import generate_query_variations
            queries = await generate_query_variations(
                question, self.cfg.retrieval.expand_n,
                model=self.cfg.generation.model, auth=self.cfg.generation.auth)
            chunks = self.retriever.retrieve_multi(queries, question)
        else:
            chunks = self.retriever.retrieve(question)
        return await self.answerer.answer(question, chunks)

    def status(self) -> dict:
        return {
            "chunks": self.store.count(),
            "sources": len(self.store.all_source_paths()),
            "index_path": self.cfg.index.path,
            "model": self.cfg.generation.model,
        }
