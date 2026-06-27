from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from .config import RaggityConfig
from .core import Raggity


class AskRequest(BaseModel):
    question: str
    expand: bool | None = None


def create_app(cfg: RaggityConfig) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["rag"] = Raggity(cfg)
        yield
        state.clear()

    app = FastAPI(title="raggity", lifespan=lifespan)

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
        answer = await rag.aask(req.question, req.expand)
        return {"answer": answer.text, "abstained": answer.abstained,
                "citations": [{"chunk_id": c.chunk_id, "source_path": c.source_path,
                               "title": c.title, "supported": c.supported}
                              for c in answer.citations]}

    return app
