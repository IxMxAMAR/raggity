"""Anthropic-style contextual retrieval: LLM-generated chunk context at ingest.

Opt-in via ``retrieval.contextual = true``. For each chunk, ONE provider call
asks the model to situate the chunk within the full document in 1-2 sentences;
that context is prepended to the chunk's stored+embedded ``text`` (citation
verification runs against this same full text, so highlighted-span overlap
still works). This is LLM-cost-heavy: one call per new/changed chunk.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import replace

from .models import Chunk

_log = logging.getLogger("raggity.contextual")

_CONTEXT_SYSTEM = (
    "You are a helpful assistant that generates brief context to situate a "
    "chunk within a larger document, for the purpose of improving search "
    "retrieval of the chunk. Respond with ONLY 1-2 short sentences of "
    "context — no preamble, no quotation marks, no restating the chunk text."
)

_CONTEXT_PROMPT_TMPL = (
    "<document>\n{doc_text}\n</document>\n"
    "Here is the chunk we want to situate within the whole document:\n"
    "<chunk>\n{chunk_text}\n</chunk>\n"
    "Please give a short succinct context to situate this chunk within the "
    "overall document for the purposes of improving search retrieval of the "
    "chunk. Answer only with the succinct context and nothing else."
)


async def _contextualize_one(doc_text: str, chunk: Chunk, provider) -> Chunk:
    prompt = _CONTEXT_PROMPT_TMPL.format(doc_text=doc_text, chunk_text=chunk.text)
    try:
        context = await provider.complete(_CONTEXT_SYSTEM, prompt)
    except Exception as exc:  # noqa: BLE001 - a single chunk failure must not abort the batch
        _log.warning(
            "contextual retrieval failed for chunk %s, leaving unchanged: %s",
            chunk.chunk_id, exc,
        )
        return chunk
    context = (context or "").strip()
    if not context:
        return chunk
    return replace(chunk, text=f"{context}\n\n{chunk.text}")


async def contextualize_chunks(
    doc_text: str, chunks: list[Chunk], provider, concurrency: int
) -> list[Chunk]:
    """Prepend an LLM-generated document-context sentence to each chunk's text.

    Runs with bounded concurrency (a semaphore, mirroring
    :func:`raggity.graph.build_graph`'s pattern) so a large document doesn't
    fan out one call per chunk unbounded. Chunk identity (``chunk_id`` and all
    other fields) is preserved — only ``text`` changes. A chunk whose provider
    call raises, or that returns a blank context, is returned unchanged rather
    than aborting the whole batch.
    """
    if not chunks:
        return []
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(chunk: Chunk) -> Chunk:
        async with sem:
            return await _contextualize_one(doc_text, chunk, provider)

    return list(await asyncio.gather(*[_one(c) for c in chunks]))
