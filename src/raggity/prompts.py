from __future__ import annotations

import re

from .models import Chunk, Citation

SYSTEM_PROMPT = (
    "You are raggity, a retrieval assistant. Answer the user's question using "
    "ONLY the provided context passages. After each factual statement, cite the "
    "passage it came from using its bracket tag exactly as given, e.g. [doc_1#abcd1234]. "
    "Only cite tags that appear in the context. If the context does not contain "
    "enough information to answer, reply exactly: "
    "\"I don't have enough information in your knowledge base to answer that.\" "
    "Do not use outside knowledge."
)

ABSTAIN_MESSAGE = (
    "I don't have enough information in your knowledge base to answer that."
)

_TAG_RE = re.compile(r"\[doc_\d+#([0-9a-f]{8})\]")
_WORD_RE = re.compile(r"[a-z0-9]+")


def chunk_tag(n: int, chunk: Chunk) -> str:
    return f"[doc_{n}#{chunk.chunk_id[:8]}]"


def format_context(chunks: list[Chunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"{chunk_tag(i, c)} ({c.source_path}):\n{c.text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[Chunk]) -> str:
    return (
        f"CONTEXT:\n{format_context(chunks)}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer with inline citations using the bracket tags above."
    )


def parse_cited_ids(answer_text: str) -> list[str]:
    return _TAG_RE.findall(answer_text)


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def verify_citations(answer_text: str, chunks: list[Chunk]) -> list[Citation]:
    by_short = {c.chunk_id[:8]: c for c in chunks}
    citations: list[Citation] = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer_text):
        for short in _TAG_RE.findall(sentence):
            chunk = by_short.get(short)
            if chunk is None:
                citations.append(Citation(chunk_id=short, source_path="?",
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
