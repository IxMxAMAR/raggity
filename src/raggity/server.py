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
        yield
        state.clear()

    app = FastAPI(title="raggity", lifespan=lifespan)

    # --- Auth dependency ---
    async def require_auth(request: Request) -> None:
        if cfg.server.auth == "none":
            return
        # auth == "api_key"
        allowed: frozenset[str] = state["allowed_keys"]
        # Check X-API-Key header
        key = request.headers.get("X-API-Key")
        if key and key in allowed:
            return
        # Check Authorization: Bearer <key>
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            bearer_key = auth_header[len("Bearer "):]
            if bearer_key in allowed:
                return
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

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

    @app.post("/ingest", dependencies=[Depends(require_auth)])
    async def ingest():
        rag: Raggity = state["rag"]
        report = await run_in_threadpool(rag.ingest)
        return {"added": report.added, "updated": report.updated,
                "deleted": report.deleted, "unchanged": report.unchanged}

    @app.get("/status", dependencies=[Depends(require_auth)])
    async def status():
        return state["rag"].status()

    @app.post("/ask", dependencies=[Depends(require_auth)])
    async def ask(req: AskRequest):
        rag: Raggity = state["rag"]
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

    @app.post("/chat", dependencies=[Depends(require_auth)])
    async def chat(req: ChatRequest):
        rag: Raggity = state["rag"]
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

    @app.delete("/session/{session_id}", status_code=204, dependencies=[Depends(require_auth)])
    async def delete_session(session_id: str):
        sessions: OrderedDict[str, Conversation] = state["sessions"]
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="session not found")
        del sessions[session_id]

    @app.get("/ask/stream", dependencies=[Depends(require_auth)])
    async def ask_stream(question: str, session_id: str | None = None):
        rag: Raggity = state["rag"]

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
