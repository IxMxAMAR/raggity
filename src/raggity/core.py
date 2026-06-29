from __future__ import annotations

import asyncio
import os


def _run_async(coro):
    """Run *coro* whether or not an event loop is already running.

    - Outside a loop (normal CLI / sync usage): ``asyncio.run()``.
    - Inside a running loop (pytest-asyncio, Jupyter): run in a new thread so
      the coroutine gets its own fresh event loop without deadlocking the outer one.

    Returns the coroutine's return value in both cases.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()  # propagate exceptions + return value
    else:
        return asyncio.run(coro)


from .answerer import ProviderAnswerer
from .config import RaggityConfig, load_config
from .conversation import Conversation
from .llm import build_provider
from .embedder import FastEmbedEmbedder
from .indexer import IngestReport, Indexer
from .models import Answer, Document
from .retriever import Retriever
from .registry import resolve
from .observability import init_tracing, span

_GRAPH_JSON = "graph.json"


class Raggity:
    def __init__(self, cfg: RaggityConfig | None = None) -> None:
        self.cfg = cfg or RaggityConfig()
        base = FastEmbedEmbedder(
            model_name=self.cfg.embedding.model,
            provider=self.cfg.embedding.provider,
            batch_size=self.cfg.embedding.batch_size,
            parallel=self.cfg.embedding.parallel,
        )
        if self.cfg.embedding.cache:
            from .cached_embedder import CachedEmbedder
            self.embedder = CachedEmbedder(base, os.path.join(self.cfg.index.path, "embed_cache.json"))
        else:
            self.embedder = base
        store_cls = resolve("store", self.cfg.index.backend)
        self.store = store_cls.from_config(self.cfg, self.embedder.dim)
        self.reranker = None
        if self.cfg.retrieval.rerank:
            from .reranker import FastEmbedReranker
            self.reranker = FastEmbedReranker(model_name=self.cfg.retrieval.rerank_model)
        self.retriever = Retriever(self.embedder, self.store, self.reranker,
                                   self.cfg.retrieval)
        self.provider = build_provider(self.cfg.generation)
        self.answerer = ProviderAnswerer(self.provider)
        # Initialise tracing (no-op when observability.tracing=False)
        init_tracing(self.cfg)
        # Lock for concurrent answer-cache load→mutate→save cycle
        self._cache_lock: asyncio.Lock | None = None
        # Graph store: lazy-load from disk when graph=true
        self._graph = None
        if self.cfg.retrieval.graph:
            graph_path = self._graph_path()
            if os.path.isfile(graph_path):
                from .graph import GraphStore
                g = GraphStore()
                g.load(graph_path)
                self._graph = g

    @classmethod
    def from_config(cls, path: str | None = None) -> "Raggity":
        return cls(load_config(path))

    @staticmethod
    def _slug(ns: str) -> str:
        """Return a filesystem/collection-safe slug for *ns*.

        Keeps alphanumerics, hyphens, and underscores.  Any run of other
        characters is replaced by an underscore.  If the result is empty or
        longer than 64 chars the sha-8 hex of the original is used instead.
        """
        import hashlib
        import re
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", ns).strip("_")
        if not slug or len(slug) > 64:
            slug = hashlib.sha256(ns.encode()).hexdigest()[:8]
        return slug

    def for_namespace(self, ns: str) -> "Raggity":
        """Return a *new* :class:`Raggity` whose index is namespaced for *ns*.

        The returned instance uses a deep copy of this config with:
        - ``index.path`` → ``<base>/users/<slug>``   (LanceDB)
        - ``index.qdrant_collection`` → ``<base_collection>_<slug>``

        The original instance (``self``) is never mutated.  Suitable for
        per-user multi-tenancy in the server.
        """
        slug = self._slug(ns)
        new_cfg = self.cfg.model_copy(deep=True)
        new_cfg.index.path = os.path.join(self.cfg.index.path, "users", slug)
        new_cfg.index.qdrant_collection = f"{self.cfg.index.qdrant_collection}_{slug}"
        return Raggity(new_cfg)

    def _manifest_path(self) -> str:
        return os.path.join(self.cfg.index.path, "manifest.json")

    def _cache_path(self) -> str:
        return os.path.join(self.cfg.index.path, "answer_cache.json")

    def _graph_path(self) -> str:
        return os.path.join(self.cfg.index.path, _GRAPH_JSON)

    def _fingerprint(self) -> str:
        rc = self.cfg.retrieval
        return (f"{self.cfg.embedding.model}|{self.embedder.dim}|"
                f"pd={rc.parent_document}|pt={rc.parent_target_tokens}|ct={rc.child_target_tokens}")

    async def build_graph(self) -> None:
        """Extract entities/relations from all indexed chunks and save graph.json.

        Requires ``cfg.retrieval.graph=true`` and a configured LLM provider.
        This is LLM-cost-heavy: one provider call per chunk in the store.
        """
        if not self.cfg.retrieval.graph:
            raise RuntimeError(
                "Graph is disabled. Set retrieval.graph = true in your config before running graph-build."
            )
        from .graph import build_graph as _build_graph
        chunks = self.store.all_chunks()
        if not chunks:
            return
        graph = await _build_graph(chunks, self.provider)
        os.makedirs(self.cfg.index.path, exist_ok=True)
        graph.save(self._graph_path())
        self._graph = graph

    def ingest(self) -> IngestReport:
        chunk_kwargs = {"parent_document": self.cfg.retrieval.parent_document,
                        "parent_target_tokens": self.cfg.retrieval.parent_target_tokens,
                        "child_target_tokens": self.cfg.retrieval.child_target_tokens}
        indexer = Indexer(self.embedder, self.store, self._manifest_path(),
                          fingerprint=self._fingerprint(), chunk_kwargs=chunk_kwargs,
                          ann_threshold=self.cfg.index.ann_threshold)
        report = indexer.ingest(self.cfg.sources.include)

        # Also ingest any configured URLs (depth=0 each, additive — no deletion)
        if self.cfg.sources.urls:
            from .connectors.web import WebConnector  # noqa: PLC0415
            docs: list[Document] = []
            for url in self.cfg.sources.urls:
                try:
                    docs.extend(WebConnector(url, depth=0).fetch())
                except Exception:
                    pass  # network errors during ingest are non-fatal
            if docs:
                report.added += self.ingest_documents(docs)

        # Build graph after vector upsert when graph=true (LLM-cost-heavy, opt-in)
        if self.cfg.retrieval.graph:
            _run_async(self.build_graph())

        return report

    def ingest_documents(self, docs: list[Document]) -> int:
        """Chunk and upsert *docs* into the index.  Returns number of docs ingested."""
        from .chunker import chunk_document  # noqa: PLC0415
        chunk_kwargs = {"parent_document": self.cfg.retrieval.parent_document,
                        "parent_target_tokens": self.cfg.retrieval.parent_target_tokens,
                        "child_target_tokens": self.cfg.retrieval.child_target_tokens}
        all_chunks = []
        for doc in docs:
            all_chunks.extend(chunk_document(doc, **chunk_kwargs))
        if all_chunks:
            self.store.upsert(all_chunks, self.embedder)
        return len(docs)

    async def _graph_neighborhood_ids(self, question: str) -> list[str]:
        """Return chunk ids from the graph neighborhood of question entities (when graph is on)."""
        if not self.cfg.retrieval.graph or self._graph is None:
            return []
        from .graph import extract
        try:
            entities, _ = await extract(question, self.provider)
        except Exception:
            return []
        nodes = self._graph.link(entities)
        ids = self._graph.neighborhood_chunk_ids(nodes, hops=self.cfg.retrieval.graph_hops)
        return list(ids)

    async def _build_queries(self, question: str, expand, hyde, step_back) -> list[str]:
        import logging as _logging
        _log = _logging.getLogger("raggity.core")
        rc = self.cfg.retrieval
        use_expand = rc.expand if expand is None else expand
        use_hyde = rc.hyde if hyde is None else hyde
        use_step = rc.step_back if step_back is None else step_back
        if use_expand:
            from .query_transform import generate_query_variations
            try:
                queries = await generate_query_variations(question, rc.expand_n, self.provider)
            except Exception as exc:
                _log.warning("query expand failed, falling back to base query: %s", exc)
                queries = [question]
        else:
            queries = [question]
        if use_hyde:
            from .query_transform import generate_hyde_document
            try:
                queries.append(await generate_hyde_document(question, self.provider))
            except Exception as exc:
                _log.warning("HyDE generation failed, skipping: %s", exc)
        if use_step:
            from .query_transform import generate_step_back_question
            try:
                queries.append(await generate_step_back_question(question, self.provider))
            except Exception as exc:
                _log.warning("step_back generation failed, skipping: %s", exc)
        return queries

    def ask(self, question: str, expand: bool | None = None,
            hyde: bool | None = None, step_back: bool | None = None,
            use_cache: bool | None = None) -> Answer:
        return _run_async(self.aask(question, expand=expand, hyde=hyde,
                                    step_back=step_back, use_cache=use_cache))

    async def aask(self, question: str, expand: bool | None = None,
                   hyde: bool | None = None, step_back: bool | None = None,
                   use_cache: bool | None = None) -> Answer:
        queries = await self._build_queries(question, expand, hyde, step_back)
        graph_ids = await self._graph_neighborhood_ids(question)
        with span("retrieve", query=question, query_count=len(queries),
                  graph_ids=len(graph_ids)):
            if queries == [question] and not graph_ids:
                chunks = self.retriever.retrieve(question)
            else:
                chunks = self.retriever.retrieve_multi(queries, question,
                                                       graph_chunk_ids=graph_ids or None)
        use_cache = self.cfg.generation.cache if use_cache is None else use_cache
        if use_cache:
            from . import cache as _cache
            from .prompts import SYSTEM_PROMPT as _SYSTEM_PROMPT
            # Lazy-create the lock inside the running loop
            if self._cache_lock is None:
                self._cache_lock = asyncio.Lock()
            key = _cache.cache_key(
                question, [c.chunk_id for c in chunks],
                self.cfg.generation.model, system_prompt=_SYSTEM_PROMPT,
            )
            async with self._cache_lock:
                data = _cache.load(self._cache_path())
                if key in data:
                    return _cache.answer_from_dict(data[key])
                with span("generate", backend=self.cfg.generation.backend,
                          model=self.cfg.generation.model, chunk_count=len(chunks)):
                    answer = await self.answerer.answer(question, chunks)
                data[key] = _cache.answer_to_dict(answer)
                _cache.save(self._cache_path(), data)
            return answer
        with span("generate", backend=self.cfg.generation.backend,
                  model=self.cfg.generation.model, chunk_count=len(chunks)):
            answer = await self.answerer.answer(question, chunks)
        return answer

    def ask_decompose(self, question: str) -> Answer:
        return _run_async(self.aask_decompose(question))

    async def aask_decompose(self, question: str) -> Answer:
        from .query_transform import decompose_question
        from .retriever import order_lost_in_middle
        subs = await decompose_question(question, self.cfg.retrieval.expand_n, self.provider)
        merged: dict[str, object] = {}
        for q in [question] + subs:
            for c in self.retriever.retrieve(q):
                merged.setdefault(c.chunk_id, c)
        pool = list(merged.values())[: self.cfg.retrieval.top_k * 2]
        # Apply reranker (if configured) then lost-in-middle ordering on merged pool.
        if self.cfg.retrieval.rerank and self.reranker is not None:
            pool = self.reranker.rerank(question, pool)
        chunks = order_lost_in_middle(pool)
        return await self.answerer.answer(question, chunks)

    async def aask_stream(self, question: str, expand: bool | None = None,
                          hyde: bool | None = None, step_back: bool | None = None,
                          use_cache: bool | None = None):
        """Yield text-delta str items then a final Answer, streaming from the answerer."""
        queries = await self._build_queries(question, expand, hyde, step_back)
        graph_ids = await self._graph_neighborhood_ids(question)
        if queries == [question] and not graph_ids:
            chunks = self.retriever.retrieve(question)
        else:
            chunks = self.retriever.retrieve_multi(queries, question,
                                                   graph_chunk_ids=graph_ids or None)
        async for piece in self.answerer.answer_stream(question, chunks):
            yield piece

    async def achat(self, conversation: Conversation, question: str) -> Answer:
        """Multi-turn chat: retrieve using history-aware query, answer with conversation context.

        Appends the user turn and the assistant answer to *conversation* before returning.
        """
        retrieval_q = conversation.retrieval_query(question)
        graph_ids = await self._graph_neighborhood_ids(retrieval_q)
        if graph_ids:
            chunks = self.retriever.retrieve_multi([retrieval_q], retrieval_q,
                                                   graph_chunk_ids=graph_ids)
        else:
            chunks = self.retriever.retrieve(retrieval_q)
        history = conversation.recent(6)
        answer = await self.answerer.answer(question, chunks, history=history or None)
        conversation.add("user", question)
        conversation.add("assistant", answer.text)
        return answer

    def chat(self, conversation: Conversation, question: str) -> Answer:
        """Synchronous wrapper for :meth:`achat`."""
        return _run_async(self.achat(conversation, question))

    def status(self) -> dict:
        return {
            "chunks": self.store.count(),
            "sources": len(self.store.all_source_paths()),
            "index_path": self.cfg.index.path,
            "model": self.cfg.generation.model,
        }
