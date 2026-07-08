from __future__ import annotations

from .chunker import estimate_tokens
from .models import Chunk


def pack_context(chunks: list[Chunk],
                 max_context_tokens: int | None) -> tuple[str, int]:
    """Greedily pack *chunks* (in retriever order) into a single context string.

    Each block is ``[source: {source_path}]\\n{text}``, joined by blank lines.
    With a budget, packing stops before the first block whose addition would push
    :func:`estimate_tokens` past *max_context_tokens*; if even the FIRST block
    does not fit it is char-truncated to ``budget * 4`` so the packed context is
    never empty while chunks exist.  Returns ``(packed, token_count)``.

    Shared by the HTTP ``POST /retrieve`` endpoint and the ``search`` MCP tool so
    both emit byte-identical ``[source: ...]`` blocks.
    """
    if not chunks:
        return "", 0
    packed = ""
    for i, c in enumerate(chunks):
        block = f"[source: {c.source_path}]\n{c.text}"
        candidate = block if not packed else f"{packed}\n\n{block}"
        if max_context_tokens is not None and estimate_tokens(candidate) > max_context_tokens:
            if i == 0:
                packed = block[: max_context_tokens * 4]
            break
        packed = candidate
    return packed, estimate_tokens(packed)
