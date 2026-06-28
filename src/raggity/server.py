from __future__ import annotations
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["rag"] = Raggity(cfg)
        state["sessions"]: dict[str, Conversation] = {}
        yield
        state.clear()

    app = FastAPI(title="raggity", lifespan=lifespan)

    def _get_or_create_session(session_id: str) -> Conversation:
        sessions: dict[str, Conversation] = state["sessions"]
        if session_id not in sessions:
            sessions[session_id] = Conversation()
        return sessions[session_id]

    @app.post("/ingest")
    async def ingest():
        rag: Raggity = state["rag"]
        report = await run_in_threadpool(rag.ingest)
        return {"added": report.added, "updated": report.updated,
                "deleted": report.deleted, "unchanged": report.unchanged}

    @app.get("/status")
    async def status():
        return state["rag"].status()

    @app.post("/ask")
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

    @app.post("/chat")
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

    @app.delete("/session/{session_id}", status_code=204)
    async def delete_session(session_id: str):
        sessions: dict[str, Conversation] = state["sessions"]
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="session not found")
        del sessions[session_id]

    @app.get("/ask/stream")
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
