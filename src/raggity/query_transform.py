from __future__ import annotations
import os
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage

_SYS = ("You generate alternative phrasings of a search query to improve retrieval. "
        "Output ONLY the alternative queries, one per line, no numbering, no commentary.")


async def generate_query_variations(question: str, n: int,
                                    model: str = "claude-opus-4-8",
                                    auth: str = "auto") -> list[str]:
    env = None
    if auth == "subscription":
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    options = ClaudeAgentOptions(system_prompt=_SYS, model=model,
                                 allowed_tools=[], permission_mode="dontAsk",
                                 **({"env": env} if env is not None else {}))
    prompt = f"Generate {n} alternative phrasings of this query:\n{question}"
    parts: list[str] = []
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                t = getattr(block, "text", None)
                if t:
                    parts.append(t)
    lines = [ln.strip() for ln in "".join(parts).splitlines() if ln.strip()]
    variations = lines[:n]
    return [question] + [v for v in variations if v != question]
