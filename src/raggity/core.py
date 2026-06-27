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

    def _cache_path(self) -> str:
        return os.path.join(self.cfg.index.path, "answer_cache.json")

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

    async def _build_queries(self, question: str, expand, hyde, step_back) -> list[str]:
        rc = self.cfg.retrieval
        use_expand = rc.expand if expand is None else expand
        use_hyde = rc.hyde if hyde is None else hyde
        use_step = rc.step_back if step_back is None else step_back
        m, a = self.cfg.generation.model, self.cfg.generation.auth
        if use_expand:
            from .query_transform import generate_query_variations
            queries = await generate_query_variations(question, rc.expand_n, model=m, auth=a)
        else:
            queries = [question]
        if use_hyde:
            from .query_transform import generate_hyde_document
            queries.append(await generate_hyde_document(question, model=m, auth=a))
        if use_step:
            from .query_transform import generate_step_back_question
            queries.append(await generate_step_back_question(question, model=m, auth=a))
        return queries

    def ask(self, question: str, expand: bool | None = None,
            hyde: bool | None = None, step_back: bool | None = None,
            use_cache: bool | None = None) -> Answer:
        return asyncio.run(self.aask(question, expand=expand, hyde=hyde,
                                     step_back=step_back, use_cache=use_cache))

    async def aask(self, question: str, expand: bool | None = None,
                   hyde: bool | None = None, step_back: bool | None = None,
                   use_cache: bool | None = None) -> Answer:
        queries = await self._build_queries(question, expand, hyde, step_back)
        if queries == [question]:
            chunks = self.retriever.retrieve(question)
        else:
            chunks = self.retriever.retrieve_multi(queries, question)
        use_cache = self.cfg.generation.cache if use_cache is None else use_cache
        key = None
        if use_cache:
            from . import cache as _cache
            data = _cache.load(self._cache_path())
            key = _cache.cache_key(question, [c.chunk_id for c in chunks], self.cfg.generation.model)
            if key in data:
                return _cache.answer_from_dict(data[key])
        answer = await self.answerer.answer(question, chunks)
        if use_cache and key is not None:
            data[key] = _cache.answer_to_dict(answer)
            _cache.save(self._cache_path(), data)
        return answer

    def ask_decompose(self, question: str) -> Answer:
        return asyncio.run(self.aask_decompose(question))

    async def aask_decompose(self, question: str) -> Answer:
        from .query_transform import decompose_question
        subs = await decompose_question(question, self.cfg.retrieval.expand_n,
                                        model=self.cfg.generation.model,
                                        auth=self.cfg.generation.auth)
        merged: dict[str, object] = {}
        for q in [question] + subs:
            for c in self.retriever.retrieve(q):
                merged.setdefault(c.chunk_id, c)
        chunks = list(merged.values())[: self.cfg.retrieval.top_k * 2]
        return await self.answerer.answer(question, chunks)

    async def aask_stream(self, question: str, expand: bool | None = None,
                          hyde: bool | None = None, step_back: bool | None = None,
                          use_cache: bool | None = None):
        """Yield text-delta str items then a final Answer, streaming from the answerer."""
        queries = await self._build_queries(question, expand, hyde, step_back)
        if queries == [question]:
            chunks = self.retriever.retrieve(question)
        else:
            chunks = self.retriever.retrieve_multi(queries, question)
        async for piece in self.answerer.answer_stream(question, chunks):
            yield piece

    def status(self) -> dict:
        return {
            "chunks": self.store.count(),
            "sources": len(self.store.all_source_paths()),
            "index_path": self.cfg.index.path,
            "model": self.cfg.generation.model,
        }
