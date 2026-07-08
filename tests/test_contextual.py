"""Tests for raggity.contextual — Anthropic-style contextual retrieval at ingest."""
from __future__ import annotations

import asyncio

import pytest

from raggity.models import Chunk


def _chunk(ordinal: int, text: str = "some chunk body") -> Chunk:
    return Chunk(
        text=text,
        source_path="doc.md",
        title="Doc",
        heading_path="Doc > Section",
        ordinal=ordinal,
        chunk_id=f"id-{ordinal}",
    )


class _FakeProvider:
    """Stub with an async .complete(system, prompt) chokepoint."""

    def __init__(self, response="A one-sentence situating context."):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


async def test_contextualize_chunks_prepends_context_to_text():
    """Each chunk's text gains 'context\\n\\n' + original text as a prefix."""
    from raggity.contextual import contextualize_chunks

    chunks = [_chunk(0, "alpha body"), _chunk(1, "beta body")]
    provider = _FakeProvider("Situating context sentence.")

    out = await contextualize_chunks("full document text", chunks, provider, concurrency=8)

    assert len(out) == 2
    for orig, new in zip(chunks, out):
        assert new.text == f"Situating context sentence.\n\n{orig.text}"
        assert new.chunk_id == orig.chunk_id  # id must stay stable
        assert new.source_path == orig.source_path
    assert len(provider.calls) == 2


async def test_contextualize_chunks_passes_doc_text_and_chunk_text_to_provider():
    from raggity.contextual import contextualize_chunks

    chunks = [_chunk(0, "the chunk body")]
    provider = _FakeProvider("ctx")

    await contextualize_chunks("THE FULL DOC", chunks, provider, concurrency=8)

    system, prompt = provider.calls[0]
    assert "THE FULL DOC" in prompt
    assert "the chunk body" in prompt
    assert isinstance(system, str) and system  # non-empty system prompt


async def test_contextualize_chunks_empty_list_returns_empty():
    from raggity.contextual import contextualize_chunks

    provider = _FakeProvider("ctx")
    out = await contextualize_chunks("doc", [], provider, concurrency=8)
    assert out == []
    assert provider.calls == []


async def test_contextualize_chunks_provider_failure_leaves_chunk_unchanged():
    """A per-chunk provider error must not abort the whole batch."""
    from raggity.contextual import contextualize_chunks

    chunks = [_chunk(0, "alpha"), _chunk(1, "beta")]

    class _FlakyProvider:
        def __init__(self):
            self.n = 0

        async def complete(self, system, prompt):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("boom")
            return "ok context"

    provider = _FlakyProvider()
    out = await contextualize_chunks("doc", chunks, provider, concurrency=8)

    assert len(out) == 2
    # First chunk's call failed -> text unchanged
    texts = {c.chunk_id: c.text for c in out}
    assert texts["id-0"] == "alpha"
    assert texts["id-1"] == "ok context\n\nbeta"


async def test_contextualize_chunks_blank_context_leaves_chunk_unchanged():
    from raggity.contextual import contextualize_chunks

    chunks = [_chunk(0, "alpha")]
    provider = _FakeProvider("   \n  ")  # whitespace-only response

    out = await contextualize_chunks("doc", chunks, provider, concurrency=8)
    assert out[0].text == "alpha"


async def test_citation_verification_works_against_contextualized_text():
    """verify_citations must still support-check against the stored
    (context-prefixed) chunk text: a sentence overlapping the ORIGINAL body
    verifies as supported, since the prefix only adds tokens."""
    from raggity.contextual import contextualize_chunks
    from raggity.prompts import chunk_tag, verify_citations

    orig = Chunk(text="backups run nightly to the NAS array",
                 source_path="ops.md", title="Ops", heading_path="Ops",
                 ordinal=0, chunk_id="b" * 64)
    provider = _FakeProvider("This section describes the backup schedule.")
    [ctx_chunk] = await contextualize_chunks("full doc", [orig], provider, concurrency=2)
    assert ctx_chunk.text.startswith("This section describes the backup schedule.\n\n")

    tag = chunk_tag(1, ctx_chunk)
    answer = f"Backups run nightly to the NAS array {tag}."
    citations = verify_citations(answer, [ctx_chunk])
    assert len(citations) == 1
    assert citations[0].supported is True
    assert citations[0].chunk_id == ctx_chunk.chunk_id


async def test_contextualize_chunks_respects_concurrency_bound():
    from raggity.contextual import contextualize_chunks

    chunks = [_chunk(i, f"body {i}") for i in range(10)]
    state = {"active": 0, "max_active": 0}
    lock = asyncio.Lock()

    class _SlowProvider:
        async def complete(self, system, prompt):
            async with lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            await asyncio.sleep(0.02)
            async with lock:
                state["active"] -= 1
            return "ctx"

    out = await contextualize_chunks("doc", chunks, _SlowProvider(), concurrency=3)

    assert len(out) == 10
    assert state["max_active"] <= 3
