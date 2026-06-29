from __future__ import annotations
import asyncio
import json
import os
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

_WEB_DIR = Path(__file__).parent / "web"

from .config import RaggityConfig
from .conversation import Conversation
from .core import Raggity
from .models import Document


class AskRequest(BaseModel):
    question: str
    expand: bool | None = None
    session_id: str | None = None


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


class IngestDocument(BaseModel):
    path: str
    text: str
    title: str | None = None


class IngestContentRequest(BaseModel):
    documents: list[IngestDocument]


def create_app(cfg: RaggityConfig) -> FastAPI:
    state: dict = {}

    # --- Resolve allowed keys: config + env (merged, deduped) ---
    def _resolve_keys() -> frozenset[str]:
        keys: list[str] = list(cfg.server.api_keys)
        env_val = os.environ.get("RAGGITY_API_KEYS", "")
        if env_val.strip():
            keys.extend(k.strip() for k in env_val.split(",") if k.strip())
        return frozenset(keys)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["rag"] = Raggity(cfg)
        # Use OrderedDict for bounded LRU session store; keys are namespaced by
        # identity: f"{identity}:{session_id}" (identity=None -> "None:...").
        state["sessions"]: OrderedDict[str, Conversation] = OrderedDict()
        state["allowed_keys"] = _resolve_keys()
        # Per-user Raggity cache: identity -> Raggity (bounded LRU, lazily built).
        state["user_rags"]: OrderedDict[str, Raggity] = OrderedDict()
        # Per-identity build locks so two concurrent cold requests don't double-build.
        state["build_locks"]: dict[str, asyncio.Lock] = {}
        # Expose for tests / introspection.
        app.state.raggity_state = state
        try:
            yield
        finally:
            # Best-effort close base rag + all cached per-user rags on teardown.
            base_rag = state.get("rag")
            user_rags = state.get("user_rags") or {}
            for rag in list(user_rags.values()):
                try:
                    await rag.close()
                except Exception:
                    pass
            if base_rag is not None:
                try:
                    await base_rag.close()
                except Exception:
                    pass
            state.clear()

    app = FastAPI(title="raggity", lifespan=lifespan)

    # --- Auth dependency ---
    # Returns the matched API key string (identity), or None when auth="none".
    async def require_auth(request: Request) -> str | None:
        if cfg.server.auth == "none":
            return None
        # auth == "api_key"
        allowed: frozenset[str] = state["allowed_keys"]
        # Check X-API-Key header
        key = request.headers.get("X-API-Key")
        if key and key in allowed:
            return key
        # Check Authorization: Bearer <key>
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            bearer_key = auth_header[len("Bearer "):]
            if bearer_key in allowed:
                return bearer_key
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    async def _resolve_rag(identity: str | None) -> Raggity:
        """Return the Raggity instance for this request.

        When ``cfg.server.per_user`` is True and auth is api_key, each identity
        gets its own namespaced :class:`Raggity` (cached in a bounded LRU).
        Otherwise the single shared base instance is returned.

        Cold per-identity construction is offloaded to a thread (model load is
        blocking) and guarded by a per-identity lock so concurrent cold requests
        build exactly once.  Eviction of the least-recently-used Raggity awaits
        its ``close()`` (best-effort).
        """
        if not cfg.server.per_user or identity is None:
            return state["rag"]
        user_rags: OrderedDict[str, Raggity] = state["user_rags"]
        if identity in user_rags:
            user_rags.move_to_end(identity)
            return user_rags[identity]
        # Cold build, guarded per-identity.
        build_locks: dict[str, asyncio.Lock] = state["build_locks"]
        lock = build_locks.get(identity)
        if lock is None:
            lock = asyncio.Lock()
            build_locks[identity] = lock
        async with lock:
            # Re-check after acquiring (another request may have built it).
            if identity in user_rags:
                user_rags.move_to_end(identity)
                return user_rags[identity]
            base_rag: Raggity = state["rag"]
            rag = await asyncio.to_thread(base_rag.for_namespace, identity)
            user_rags[identity] = rag
            # Evict LRU on overflow.
            while len(user_rags) > cfg.server.max_user_rags:
                _evicted_key, evicted = user_rags.popitem(last=False)
                build_locks.pop(_evicted_key, None)
                try:
                    await evicted.close()
                except Exception:
                    pass
            return rag

    def _get_or_create_session(identity: str | None, session_id: str) -> Conversation:
        """Return the caller's session, namespaced by identity.

        The store key is ``f"{identity}:{session_id}"`` so two identities can use
        the same ``session_id`` without colliding or reading each other's history.
        """
        key = f"{identity}:{session_id}"
        sessions: OrderedDict[str, Conversation] = state["sessions"]
        if key in sessions:
            sessions.move_to_end(key)
        else:
            sessions[key] = Conversation()
            if len(sessions) > cfg.server.max_sessions:
                sessions.popitem(last=False)
        return sessions[key]

    @app.post("/ingest")
    async def ingest(identity: str | None = Depends(require_auth)):
        rag: Raggity = await _resolve_rag(identity)
        report = await asyncio.to_thread(rag.ingest)
        return {"added": report.added, "updated": report.updated,
                "deleted": report.deleted, "unchanged": report.unchanged}

    @app.post("/ingest/content")
    async def ingest_content(req: IngestContentRequest,
                             identity: str | None = Depends(require_auth)):
        rag: Raggity = await _resolve_rag(identity)
        docs = [
            Document(path=d.path, title=d.title or d.path, text=d.text,
                     file_hash="", mtime=0.0)
            for d in req.documents
        ]
        n = await asyncio.to_thread(rag.ingest_documents, docs)
        return {"ingested": n}

    @app.get("/status")
    async def status(identity: str | None = Depends(require_auth)):
        rag: Raggity = await _resolve_rag(identity)
        return await asyncio.to_thread(rag.status)

    @app.post("/ask")
    async def ask(req: AskRequest, identity: str | None = Depends(require_auth)):
        rag: Raggity = await _resolve_rag(identity)
        if req.session_id is not None:
            conv = _get_or_create_session(identity, req.session_id)
            answer = await rag.achat(conv, req.question)
            return {
                "answer": answer.text,
                "abstained": answer.abstained,
                "citations": [
                    {"chunk_id": c.chunk_id, "source_path": c.source_path,
                     "title": c.title, "supported": c.supported}
                    for c in answer.citations
                ],
                "session_id": req.session_id,
            }
        answer = await rag.aask(req.question, req.expand)
        return {
            "answer": answer.text,
            "abstained": answer.abstained,
            "citations": [
                {"chunk_id": c.chunk_id, "source_path": c.source_path,
                 "title": c.title, "supported": c.supported}
                for c in answer.citations
            ],
        }

    @app.post("/chat")
    async def chat(req: ChatRequest, identity: str | None = Depends(require_auth)):
        rag: Raggity = await _resolve_rag(identity)
        session_id = req.session_id if req.session_id is not None else uuid.uuid4().hex
        conv = _get_or_create_session(identity, session_id)
        answer = await rag.achat(conv, req.question)
        return {
            "answer": answer.text,
            "abstained": answer.abstained,
            "citations": [
                {"chunk_id": c.chunk_id, "source_path": c.source_path,
                 "title": c.title, "supported": c.supported}
                for c in answer.citations
            ],
            "session_id": session_id,
        }

    @app.delete("/session/{session_id}", status_code=204)
    async def delete_session(session_id: str, identity: str | None = Depends(require_auth)):
        sessions: OrderedDict[str, Conversation] = state["sessions"]
        key = f"{identity}:{session_id}"
        # Only the caller's own namespaced session may be deleted; another
        # identity's session is invisible (404), so it cannot be probed/removed.
        if key not in sessions:
            raise HTTPException(status_code=404, detail="session not found")
        del sessions[key]

    @app.get("/ask/stream")
    async def ask_stream(question: str, session_id: str | None = None,
                         identity: str | None = Depends(require_auth)):
        rag: Raggity = await _resolve_rag(identity)

        async def _event_stream() -> AsyncIterator[str]:
            from .models import Answer  # noqa: PLC0415
            try:
                if session_id is not None:
                    # Session-aware streaming: collect via achat then yield as single chunk
                    # (achat is not a generator but we can still stream the answer text)
                    conv = _get_or_create_session(identity, session_id)
                    answer: Answer = await rag.achat(conv, question)
                    yield f"data: {answer.text}\n\n"
                    done_payload = {
                        "citations": [
                            {"chunk_id": c.chunk_id, "source_path": c.source_path,
                             "title": c.title, "supported": c.supported}
                            for c in answer.citations
                        ],
                        "session_id": session_id,
                    }
                    yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
                else:
                    # Stateless streaming via aask_stream
                    final_answer: Answer | None = None
                    async for piece in rag.aask_stream(question):
                        if isinstance(piece, Answer):
                            final_answer = piece
                        else:
                            yield f"data: {piece}\n\n"
                    if final_answer is not None:
                        done_payload = {
                            "citations": [
                                {"chunk_id": c.chunk_id, "source_path": c.source_path,
                                 "title": c.title, "supported": c.supported}
                                for c in final_answer.citations
                            ],
                        }
                    else:
                        done_payload = {"citations": []}
                    yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
            except asyncio.CancelledError:
                # Client disconnected — stop cleanly without emitting anything.
                return
            except Exception:
                # Never leak internals/tracebacks to the client.
                yield 'event: error\ndata: {"error": "internal error"}\n\n'
                return

        return StreamingResponse(_event_stream(), media_type="text/event-stream")

    @app.get("/")
    async def root():
        return FileResponse(_WEB_DIR / "index.html", media_type="text/html")

    return app
