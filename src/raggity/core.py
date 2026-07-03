from __future__ import annotations

import asyncio
import json
import os
import threading


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

# Sentinel for un-built lazy component slots.  A dedicated object (not ``None``)
# so that legitimately-``None`` components (e.g. a disabled reranker) are cached
# and not rebuilt on every access.
_UNSET = object()


class Raggity:
    def __init__(self, cfg: RaggityConfig | None = None, *,
                 _shared_from: "Raggity | None" = None) -> None:
        self.cfg = cfg or RaggityConfig()
        # When set, heavy *stateless* models (raw embedder, reranker, LLM
        # provider) are borrowed from this base instance instead of rebuilt —
        # onnxruntime InferenceSession.run() is thread-safe, so one model can be
        # shared across tenant threads.  The store and the (per-tenant) cached
        # embedder wrapper are always this instance's own.
        self._shared = _shared_from
        # RLock (not Lock): the retriever factory re-enters to build embedder /
        # store / reranker while already holding the lock.
        self._build_lock = threading.RLock()
        self._raw_embedder = _UNSET
        self._embedder = _UNSET
        self._store = _UNSET
        self._reranker = _UNSET
        self._provider = _UNSET
        self._answerer = _UNSET
        self._retriever = _UNSET

        # Initialise tracing (no-op when observability.tracing=False) — eager.
        init_tracing(self.cfg)
        # Lock for concurrent answer-cache load→mutate→save cycle
        self._cache_lock: asyncio.Lock | None = None
        # Graph store: eager load-on-init from disk when graph=true.
        self._graph = None
        if self.cfg.retrieval.graph:
            graph_path = self._graph_path()
            if os.path.isfile(graph_path):
                from .graph import GraphStore
                g = GraphStore()
                g.load(graph_path)
                self._graph = g

    # -- lazy component construction --------------------------------------
    def _lazy(self, slot: str, factory):
        """Double-checked lazy build of ``self.<slot>`` under ``_build_lock``."""
        cur = getattr(self, slot)
        if cur is not _UNSET:
            return cur
        with self._build_lock:
            cur = getattr(self, slot)
            if cur is not _UNSET:
                return cur
            val = factory()
            setattr(self, slot, val)
            return val

    @property
    def raw_embedder(self):
        def _factory():
            if self._shared is not None:
                return self._shared.raw_embedder
            return FastEmbedEmbedder(
                model_name=self.cfg.embedding.model,
                provider=self.cfg.embedding.provider,
                batch_size=self.cfg.embedding.batch_size,
                parallel=self.cfg.embedding.parallel,
            )
        return self._lazy("_raw_embedder", _factory)

    @property
    def embedder(self):
        def _factory():
            raw = self.raw_embedder
            if self.cfg.embedding.cache:
                from .cached_embedder import CachedEmbedder
                return CachedEmbedder(
                    raw, os.path.join(self.cfg.index.path, "embed_cache.sqlite"))
            return raw
        return self._lazy("_embedder", _factory)

    @property
    def store(self):
        def _factory():
            store_cls = resolve("store", self.cfg.index.backend)
            # Pass dim as a callable so an existing index opens without building
            # the embedder just to learn its dimension.
            return store_cls.from_config(self.cfg, lambda: self.embedder.dim)
        return self._lazy("_store", _factory)

    @property
    def reranker(self):
        def _factory():
            if not self.cfg.retrieval.rerank:
                return None
            if self._shared is not None:
                return self._shared.reranker
            from .reranker import FastEmbedReranker
            return FastEmbedReranker(model_name=self.cfg.retrieval.rerank_model)
        return self._lazy("_reranker", _factory)

    @property
    def provider(self):
        def _factory():
            if self._shared is not None:
                return self._shared.provider
            return build_provider(self.cfg.generation)
        return self._lazy("_provider", _factory)

    @property
    def answerer(self):
        return self._lazy("_answerer", lambda: ProviderAnswerer(self.provider))

    @property
    def retriever(self):
        return self._lazy(
            "_retriever",
            lambda: Retriever(self.embedder, self.store, self.reranker,
                              self.cfg.retrieval),
        )

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
        # Share heavy models with this base instance (cheap tenant construction).
        return Raggity(new_cfg, _shared_from=self)

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
        graph = await _build_graph(chunks, self.provider,
                                   concurrency=self.cfg.retrieval.graph_concurrency)
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

        # Also ingest any configured URLs. depth=0 fetches exactly one page, so no
        # in-scope page can have vanished -> scope=None (a raw-URL prefix scope
        # could wrongly prune other URLs that merely share the string prefix).
        if self.cfg.sources.urls:
            from .connectors.web import WebConnector  # noqa: PLC0415
            for url in self.cfg.sources.urls:
                try:
                    docs = WebConnector(url, depth=0).fetch()
                except Exception:
                    docs = []  # network errors during ingest are non-fatal
                if docs:
                    report.added += self.ingest_documents(docs, scope=None)

        # Build graph after vector upsert when graph=true (LLM-cost-heavy, opt-in)
        if self.cfg.retrieval.graph:
            _run_async(self.build_graph())

        return report

    def _connector_manifest_path(self) -> str:
        return os.path.join(self.cfg.index.path, "manifest_connectors.json")

    def _load_connector_manifest(self) -> dict[str, dict]:
        p = self._connector_manifest_path()
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _save_connector_manifest(self, manifest: dict[str, dict]) -> None:
        os.makedirs(self.cfg.index.path, exist_ok=True)
        p = self._connector_manifest_path()
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        os.replace(tmp, p)

    def ingest_documents(self, docs: list[Document], scope: str | None = None) -> int:
        """Chunk and upsert *docs* into the index (connector-incremental).

        A separate ``manifest_connectors.json`` records each doc's content hash.
        Docs whose hash matches are skipped (no re-chunk/re-embed).  A doc with
        an empty ``file_hash`` (e.g. server ``/ingest/content``) never matches
        and is always upserted.  When *scope* is given (a delimiter-terminated
        prefix over ``doc.path``), entries under that scope that were NOT seen
        in this call are treated as vanished and their chunks removed — deletion
        is confined to the scope, leaving other scopes intact.  Returns the
        number of docs passed in (unchanged return semantics).
        """
        from .chunker import chunk_document  # noqa: PLC0415
        chunk_kwargs = {"parent_document": self.cfg.retrieval.parent_document,
                        "parent_target_tokens": self.cfg.retrieval.parent_target_tokens,
                        "child_target_tokens": self.cfg.retrieval.child_target_tokens}
        manifest = self._load_connector_manifest()
        seen: set[str] = set()
        to_delete: set[str] = set()
        all_chunks = []
        for doc in docs:
            seen.add(doc.path)
            prev = manifest.get(doc.path)
            if doc.file_hash and prev is not None and prev.get("hash") == doc.file_hash:
                continue  # content unchanged — skip re-chunk/re-embed
            all_chunks.extend(chunk_document(doc, **chunk_kwargs))
            manifest[doc.path] = {"hash": doc.file_hash}
        if scope is not None:
            for path in list(manifest.keys()):
                if path.startswith(scope) and path not in seen:
                    to_delete.add(path)
                    del manifest[path]
        if to_delete:
            self.store.delete_sources(list(to_delete))
        if all_chunks:
            self.store.upsert(all_chunks, self.embedder)
        self._save_connector_manifest(manifest)
        return len(docs)

    async def _cached_transform(self, kind: str, question: str, n: int,
                                use_cache: bool, compute):
        """Return a query-transform / graph-extract output, optionally cached.

        When *use_cache* is set the result is memoised in the answer-cache JSON
        file under a ``tf:`` key (see :func:`cache.transform_key`) using the same
        reload-merge-under-lock discipline as answer caching, so a repeat of the
        same question skips the provider call.  *compute* is an async callable
        returning a JSON-serialisable payload dict.
        """
        if not use_cache:
            return await compute()
        from . import cache as _cache
        if self._cache_lock is None:
            self._cache_lock = asyncio.Lock()
        key = _cache.transform_key(kind, question, self.cfg.generation.model, n)
        async with self._cache_lock:
            data = _cache.load(self._cache_path())
            if key in data:
                return data[key]
        value = await compute()
        async with self._cache_lock:
            data = _cache.load(self._cache_path())
            data[key] = value
            _cache.save(self._cache_path(), data)
        return value

    async def _graph_neighborhood_ids(self, question: str,
                                      use_cache: bool = False) -> list[str]:
        """Return chunk ids from the graph neighborhood of question entities (when graph is on)."""
        if not self.cfg.retrieval.graph or self._graph is None:
            return []
        from .graph import extract

        async def _compute():
            entities, _ = await extract(question, self.provider)
            return {"entities": entities}

        try:
            payload = await self._cached_transform(
                "graph_entities", question, 0, use_cache, _compute)
        except Exception:
            return []
        entities = payload.get("entities", [])
        nodes = self._graph.link(entities)
        ids = self._graph.neighborhood_chunk_ids(nodes, hops=self.cfg.retrieval.graph_hops)
        return list(ids)

    async def _build_queries(self, question: str, expand, hyde, step_back,
                             use_cache: bool = False) -> list[str]:
        import logging as _logging
        _log = _logging.getLogger("raggity.core")
        rc = self.cfg.retrieval
        use_expand = rc.expand if expand is None else expand
        use_hyde = rc.hyde if hyde is None else hyde
        use_step = rc.step_back if step_back is None else step_back

        async def _do_expand() -> list[str]:
            if not use_expand:
                return [question]
            from .query_transform import generate_query_variations

            async def _compute():
                qs = await generate_query_variations(question, rc.expand_n, self.provider)
                return {"queries": qs}

            try:
                payload = await self._cached_transform(
                    "expand", question, rc.expand_n, use_cache, _compute)
                return payload.get("queries") or [question]
            except Exception as exc:
                _log.warning("query expand failed, falling back to base query: %s", exc)
                return [question]

        async def _do_hyde() -> str | None:
            if not use_hyde:
                return None
            from .query_transform import generate_hyde_document

            async def _compute():
                return {"text": await generate_hyde_document(question, self.provider)}

            try:
                payload = await self._cached_transform(
                    "hyde", question, 0, use_cache, _compute)
                return payload.get("text")
            except Exception as exc:
                _log.warning("HyDE generation failed, skipping: %s", exc)
                return None

        async def _do_step() -> str | None:
            if not use_step:
                return None
            from .query_transform import generate_step_back_question

            async def _compute():
                return {"text": await generate_step_back_question(question, self.provider)}

            try:
                payload = await self._cached_transform(
                    "step_back", question, 0, use_cache, _compute)
                return payload.get("text")
            except Exception as exc:
                _log.warning("step_back generation failed, skipping: %s", exc)
                return None

        # Enabled transforms run concurrently; each carries its own fallback so
        # gather never sees an exception.
        expand_qs, hyde_text, step_text = await asyncio.gather(
            _do_expand(), _do_hyde(), _do_step())
        queries = list(expand_qs)
        if hyde_text is not None:
            queries.append(hyde_text)
        if step_text is not None:
            queries.append(step_text)
        return queries

    def ask(self, question: str, expand: bool | None = None,
            hyde: bool | None = None, step_back: bool | None = None,
            use_cache: bool | None = None) -> Answer:
        return _run_async(self.aask(question, expand=expand, hyde=hyde,
                                    step_back=step_back, use_cache=use_cache))

    async def aask(self, question: str, expand: bool | None = None,
                   hyde: bool | None = None, step_back: bool | None = None,
                   use_cache: bool | None = None) -> Answer:
        use_cache = self.cfg.generation.cache if use_cache is None else use_cache
        # Concurrent prelude: query transforms + graph neighborhood run together.
        queries, graph_ids = await asyncio.gather(
            self._build_queries(question, expand, hyde, step_back, use_cache=use_cache),
            self._graph_neighborhood_ids(question, use_cache=use_cache),
        )
        with span("retrieve", query=question, query_count=len(queries),
                  graph_ids=len(graph_ids)):
            if queries == [question] and not graph_ids:
                chunks = await asyncio.to_thread(self.retriever.retrieve, question)
            else:
                chunks = await asyncio.to_thread(
                    self.retriever.retrieve_multi, queries, question,
                    graph_chunk_ids=graph_ids or None)
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
            # Narrowed lock: check under lock, GENERATE OUTSIDE the lock (so
            # concurrent distinct-question generations do not serialize), then
            # reload-merge-save under the lock again (idempotent overwrite).
            async with self._cache_lock:
                data = _cache.load(self._cache_path())
                if key in data:
                    return _cache.answer_from_dict(data[key])
            with span("generate", backend=self.cfg.generation.backend,
                      model=self.cfg.generation.model, chunk_count=len(chunks)):
                answer = await self.answerer.answer(question, chunks)
            async with self._cache_lock:
                data = _cache.load(self._cache_path())
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
            for c in await asyncio.to_thread(self.retriever.retrieve, q):
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
        use_cache = self.cfg.generation.cache if use_cache is None else use_cache
        queries, graph_ids = await asyncio.gather(
            self._build_queries(question, expand, hyde, step_back, use_cache=use_cache),
            self._graph_neighborhood_ids(question, use_cache=use_cache),
        )
        if queries == [question] and not graph_ids:
            chunks = await asyncio.to_thread(self.retriever.retrieve, question)
        else:
            chunks = await asyncio.to_thread(
                self.retriever.retrieve_multi, queries, question,
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
            chunks = await asyncio.to_thread(
                self.retriever.retrieve_multi, [retrieval_q], retrieval_q,
                graph_chunk_ids=graph_ids)
        else:
            chunks = await asyncio.to_thread(self.retriever.retrieve, retrieval_q)
        history = conversation.recent(6)
        answer = await self.answerer.answer(question, chunks, history=history or None)
        conversation.add("user", question)
        conversation.add("assistant", answer.text)
        return answer

    async def achat_stream(self, conversation: Conversation, question: str):
        """Stream a multi-turn chat answer: yield text deltas, then the final Answer.

        Retrieval is history-aware (same query as :meth:`achat`).  Both the user
        turn and the assistant answer are appended to *conversation* ONLY after
        the final :class:`Answer` arrives — so a client disconnect mid-stream
        (which surfaces as ``CancelledError``/``GeneratorExit`` and is
        deliberately NOT caught here) leaves no half-recorded turn.
        """
        retrieval_q = conversation.retrieval_query(question)
        graph_ids = await self._graph_neighborhood_ids(retrieval_q)
        if graph_ids:
            chunks = await asyncio.to_thread(
                self.retriever.retrieve_multi, [retrieval_q], retrieval_q,
                graph_chunk_ids=graph_ids)
        else:
            chunks = await asyncio.to_thread(self.retriever.retrieve, retrieval_q)
        history = conversation.recent(6)
        final: Answer | None = None
        async for piece in self.answerer.answer_stream(
                question, chunks, history=history or None):
            if isinstance(piece, Answer):
                final = piece
            else:
                yield piece
        if final is not None:
            conversation.add("user", question)
            conversation.add("assistant", final.text)
            yield final

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

    async def close(self) -> None:
        """Release underlying resources (best-effort).

        Reads component slots directly (never the lazy properties) so closing an
        instance never *builds* anything.  Ownership rules:

        - The store and the (per-instance) cached-embedder connection are always
          this instance's own → closed here.
        - The LLM provider is closed only by a root instance (``_shared`` is
          ``None``); a tenant borrows the base's provider and must not close it.
        - Shared raw embedder / reranker are never closed by anyone here.
        """
        # Provider: root-only (tenants share the base's provider).
        if self._shared is None:
            provider = self.__dict__.get("_provider", _UNSET)
            if provider is not None and provider is not _UNSET:
                try:
                    await provider.aclose()
                except Exception:
                    pass
        # Store: always own.
        store = self.__dict__.get("_store", _UNSET)
        if store is not None and store is not _UNSET:
            try:
                aclose = getattr(store, "aclose", None)
                if aclose is not None:
                    await aclose()
                else:
                    close = getattr(store, "close", None)
                    if close is not None:
                        close()
                    else:
                        exit_ = getattr(store, "__exit__", None)
                        if exit_ is not None:
                            exit_(None, None, None)
            except Exception:
                pass
        # Cached-embedder sqlite connection: per-instance, safe to close.
        embedder = self.__dict__.get("_embedder", _UNSET)
        if embedder is not None and embedder is not _UNSET:
            closer = getattr(embedder, "close", None)
            if closer is not None:
                try:
                    closer()
                except Exception:
                    pass
