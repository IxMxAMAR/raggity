from __future__ import annotations

import logging
import re

from .models import Chunk, Citation

log = logging.getLogger("raggity.prompts")

# Number of hex characters used in citation tags.
# 16 hex chars = 64 bits of chunk_id prefix — collision probability negligible
# for contexts up to thousands of chunks (birthday bound ~2^32 for 8-char).
_TAG_PREFIX_LEN = 16

SYSTEM_PROMPT = (
    "You are raggity, a retrieval assistant. Answer the user's question using "
    "ONLY the provided context passages. After each factual statement, cite the "
    "passage it came from using its bracket tag exactly as given, "
    f"e.g. [doc_1#{'a' * _TAG_PREFIX_LEN}]. "
    "Only cite tags that appear in the context. If the context does not contain "
    "enough information to answer, reply exactly: "
    "\"I don't have enough information in your knowledge base to answer that.\" "
    "Do not use outside knowledge."
)

ABSTAIN_MESSAGE = (
    "I don't have enough information in your knowledge base to answer that."
)

_TAG_RE = re.compile(rf"\[doc_\d+#([0-9a-f]{{{_TAG_PREFIX_LEN}}})\]")
_WORD_RE = re.compile(r"[a-z0-9]+")


def chunk_tag(n: int, chunk: Chunk) -> str:
    return f"[doc_{n}#{chunk.chunk_id[:_TAG_PREFIX_LEN]}]"


def format_context(chunks: list[Chunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"{chunk_tag(i, c)} ({c.source_path}):\n{c.text}")
    return "\n\n".join(blocks)


def build_user_prompt(
    question: str,
    chunks: list[Chunk],
    history: list[tuple[str, str]] | None = None,
) -> str:
    prefix = ""
    if history:
        lines = "\n".join(f"{role}: {text}" for role, text in history)
        prefix = f"CONVERSATION SO FAR:\n{lines}\n\n"
    return (
        f"{prefix}"
        f"CONTEXT:\n{format_context(chunks)}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer with inline citations using the bracket tags above."
    )


def parse_cited_ids(answer_text: str) -> list[str]:
    return _TAG_RE.findall(answer_text)


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def verify_citations(answer_text: str, chunks: list[Chunk]) -> list[Citation]:
    # Build prefix→chunk map; detect and log collisions among in-context chunks.
    by_prefix: dict[str, Chunk] = {}
    for c in chunks:
        prefix = c.chunk_id[:_TAG_PREFIX_LEN]
        if prefix in by_prefix:
            log.warning(
                "raggity.prompts: citation-tag prefix collision detected "
                "(%s): chunk_ids %r and %r share the same %d-char prefix — "
                "second chunk skipped in citation lookup.",
                prefix, by_prefix[prefix].chunk_id, c.chunk_id, _TAG_PREFIX_LEN,
            )
        else:
            by_prefix[prefix] = c

    citations: list[Citation] = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer_text):
        for prefix in _TAG_RE.findall(sentence):
            chunk = by_prefix.get(prefix)
            if chunk is None:
                citations.append(Citation(chunk_id=prefix, source_path="?",
                                          title="?", supported=False))
                continue
            sent_tokens = _tokens(sentence) - {"doc"}
            chunk_tokens = _tokens(chunk.text)
            overlap = len(sent_tokens & chunk_tokens) / max(1, len(sent_tokens))
            citations.append(Citation(
                chunk_id=chunk.chunk_id, source_path=chunk.source_path,
                title=chunk.title, supported=overlap >= 0.25,
            ))
    return citations
