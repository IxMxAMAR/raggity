"""Agentic multi-round retrieval: the Claude model orchestrates its own searches.

`ask_agentic` gives the model a single in-process SDK MCP tool,
``search_knowledge_base(query, k)``, and lets it search-read-refine over several
rounds before answering. Every chunk retrieved across every round is accumulated
into one pool; the final answer's citations are verified against that whole pool.

Claude backend ONLY — the mechanism is the Agent SDK's in-process tool callback
(``create_sdk_mcp_server`` / ``@tool``). For non-Claude backends, use
``retrieval.corrective`` for a bounded corrective loop instead.

The llm chokepoint (``raggity.llm.query`` + ``AssistantMessage``) is referenced
via the ``llm`` module namespace so tests patch it exactly as elsewhere. The tool
and server builders are the guarded SDK globals filled by ``llm._ensure_sdk()``.
"""
from __future__ import annotations

import asyncio

from . import llm
from .models import Answer, Chunk
from .prompts import ABSTAIN_MESSAGE, chunk_tag, verify_citations

# Server + tool names. The tool's full name (as the CLI/agent addresses it) is
# ``mcp__<server>__<tool>`` — this is what goes in allowed_tools.
KB_SERVER_NAME = "raggity_kb"
KB_TOOL_NAME = "search_knowledge_base"
KB_TOOL_FULLNAME = f"mcp__{KB_SERVER_NAME}__{KB_TOOL_NAME}"

AGENTIC_SYSTEM_PROMPT = (
    "You are raggity, a retrieval assistant answering from the user's knowledge "
    "base. You have one tool, search_knowledge_base(query, k), which returns "
    "passages, each prefixed with a bracket citation tag like "
    "[doc_1#0123456789abcdef] and a [source: ...] line.\n\n"
    "Work iteratively: search with a focused query, read the passages, and if "
    "they leave gaps or hint at related terms, search again with different "
    "phrasings or follow-up queries (typically 2-4 searches, a few more only if "
    "genuinely needed). Then answer the user's question using ONLY the retrieved "
    "passages. After each factual statement, cite the passage it came from using "
    "its bracket tag exactly as given. Only cite tags that actually appeared in "
    "tool results. Do not use outside knowledge.\n\n"
    "If, after searching, the passages do not contain enough information to "
    "answer, reply with EXACTLY this sentence and nothing else: "
    f"\"{ABSTAIN_MESSAGE}\""
)

# JSON schema keeps ``k`` optional (create_sdk_mcp_server would otherwise mark
# every property required); the handler defaults it to 6.
_KB_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string",
                  "description": "Search query for the knowledge base."},
        "k": {"type": "integer",
              "description": "Number of passages to return (default 6)."},
    },
    "required": ["query"],
}


def _pack(chunks: list[Chunk], tag_index) -> str:
    """Render retrieved *chunks* with the citation tags the answerer expects."""
    if not chunks:
        return "(no passages found for that query)"
    blocks = []
    for c in chunks:
        tag = chunk_tag(tag_index(c.chunk_id), c)
        blocks.append(f"{tag}\n[source: {c.source_path}]\n{c.text}")
    return "\n\n".join(blocks)


async def ask_agentic(rag, question: str) -> Answer:
    """Answer *question* via model-orchestrated multi-round retrieval.

    Raises ``RuntimeError`` when the generation backend is not Claude.
    """
    cfg = rag.cfg
    if cfg.generation.backend != "claude":
        raise RuntimeError(
            f"--agentic requires generation.backend='claude' (got "
            f"{cfg.generation.backend!r}); the mechanism is the Claude Agent SDK's "
            "in-process tool. For other backends, enable retrieval.corrective for a "
            "corrective retrieval loop instead."
        )

    # Accumulated pool across all rounds, keyed by chunk_id; `order` gives each
    # chunk a stable citation-tag number the first time it is seen.
    accumulated: dict[str, Chunk] = {}
    order: list[str] = []

    def _tag_index(chunk_id: str) -> int:
        return order.index(chunk_id) + 1

    async def _search_handler(args: dict) -> dict:
        query_text = args["query"]
        k = args.get("k") or 6
        chunks = await asyncio.to_thread(
            rag.retriever.retrieve, query_text, top_k=k, apply_sufficiency=False)
        for c in chunks:
            if c.chunk_id not in accumulated:
                order.append(c.chunk_id)
                accumulated[c.chunk_id] = c
        return {"content": [{"type": "text", "text": _pack(chunks, _tag_index)}]}

    llm._ensure_sdk()
    search_tool = llm.tool(KB_TOOL_NAME,
                           "Search the user's knowledge base and return relevant "
                           "passages, each with a citation tag to cite in your answer.",
                           _KB_INPUT_SCHEMA)(_search_handler)
    server = llm.create_sdk_mcp_server(name=KB_SERVER_NAME, tools=[search_tool])
    # Stash the tool on the in-process server instance so tests can drive the
    # handler directly (the real SDK routes through its own registered handlers
    # and never reads this attribute).
    try:
        server["instance"]._raggity_tools = [search_tool]
    except Exception:  # noqa: BLE001 - reachability aid only
        pass

    kw = llm.base_options_kwargs(AGENTIC_SYSTEM_PROMPT, cfg.generation.model,
                                 cfg.generation.auth)
    kw.update(mcp_servers={KB_SERVER_NAME: server},
              allowed_tools=[KB_TOOL_FULLNAME], max_turns=8)
    opts = llm.ClaudeAgentOptions(**kw)

    # Collect assistant text; the answer is the LAST assistant message's text
    # (intermediate turns carry tool-use / reasoning). Mirrors ClaudeProvider's
    # AssistantMessage handling.
    final_text = ""
    async for message in llm.query(prompt=question, options=opts):
        if llm.AssistantMessage is not None and isinstance(message, llm.AssistantMessage):
            parts = [getattr(b, "text", None) for b in message.content]
            joined = "".join(p for p in parts if p)
            if joined:
                final_text = joined
    final_text = final_text.strip()

    all_chunks = [accumulated[cid] for cid in order]
    if not final_text or final_text == ABSTAIN_MESSAGE:
        return Answer(text=final_text or ABSTAIN_MESSAGE, citations=[], abstained=True)
    return Answer(text=final_text,
                  citations=verify_citations(final_text, all_chunks),
                  abstained=False)
