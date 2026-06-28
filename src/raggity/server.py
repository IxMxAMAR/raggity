from __future__ import annotations
import json
import os
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

_WEB_DIR = Path(__file__).parent / "web"

from .config import RaggityConfig
from .conversation import Conversation
from .core import Raggity


class AskRequest(BaseModel):
    question: str
    expand: bool | None = None
    session_id: str | None = None


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


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
        # Use OrderedDict for bounded LRU session store
        state["sessions"]: OrderedDict[str, Conversation] = OrderedDict()
        state["allowed_keys"] = _resolve_keys()
        # Per-user Raggity cache: ns -> Raggity (created lazily, cached for reuse)
        state["user_rags"]: dict[str, Raggity] = {}
        yield
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

    def _resolve_rag(identity: str | None) -> Raggity:
        """Return the Raggity instance for this request.

        When ``cfg.server.per_user`` is True and auth is api_key, each identity
        gets its own namespaced :class:`Raggity` (cached for reuse).  Otherwise
        the single shared base instance is returned.
        """
        if not cfg.server.per_user or identity is None:
            return state["rag"]
        user_rags: dict[str, Raggity] = state["user_rags"]
        if identity not in user_rags:
            base_rag: Raggity = state["rag"]
            user_rags[identity] = base_rag.for_namespace(identity)
        return user_rags[identity]

    def _get_or_create_session(session_id: str) -> Conversation:
        sessions: OrderedDict[str, Conversation] = state["sessions"]
        if session_id in sessions:
            # Move to end (most recently used)
            sessions.move_to_end(session_id)
        else:
            sessions[session_id] = Conversation()
            # Evict oldest if over capacity
            if len(sessions) > cfg.server.max_sessions:
                sessions.popitem(last=False)
        return sessions[session_id]

    @app.post("/ingest")
    async def ingest(identity: str | None = Depends(require_auth)):
        rag: Raggity = _resolve_rag(identity)
        report = await run_in_threadpool(rag.ingest)
        return {"added": report.added, "updated": report.updated,
                "deleted": report.deleted, "unchanged": report.unchanged}

    @app.get("/status")
    async def status(identity: str | None = Depends(require_auth)):
        return _resolve_rag(identity).status()

    @app.post("/ask")
    async def ask(req: AskRequest, identity: str | None = Depends(require_auth)):
        rag: Raggity = _resolve_rag(identity)
        if req.session_id is not None:
            conv = _get_or_create_session(req.session_id)
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
        rag: Raggity = _resolve_rag(identity)
        session_id = req.session_id if req.session_id is not None else uuid.uuid4().hex
        conv = _get_or_create_session(session_id)
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
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="session not found")
        del sessions[session_id]

    @app.get("/ask/stream")
    async def ask_stream(question: str, session_id: str | None = None,
                         identity: str | None = Depends(require_auth)):
        rag: Raggity = _resolve_rag(identity)

        async def _event_stream() -> AsyncIterator[str]:
            from .models import Answer  # noqa: PLC0415
            if session_id is not None:
                # Session-aware streaming: collect via achat then yield as single chunk
                # (achat is not a generator but we can still stream the answer text)
                conv = _get_or_create_session(session_id)
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
                from .models import Answer  # noqa: PLC0415 # re-import fine
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

        return StreamingResponse(_event_stream(), media_type="text/event-stream")

    @app.get("/")
    async def root():
        return FileResponse(_WEB_DIR / "index.html", media_type="text/html")

    return app
