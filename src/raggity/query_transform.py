from __future__ import annotations
import os
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage

_SYS = ("You generate alternative phrasings of a search query to improve retrieval. "
        "Output ONLY the alternative queries, one per line, no numbering, no commentary.")

_HYDE_SYS = ("Write a short, plausible answer passage for the user's question, as if "
             "excerpted from a relevant document. Output only the passage, no preamble.")

_STEPBACK_SYS = ("Given a specific question, produce ONE broader, more general question "
                 "that would surface useful background. Output only the question.")


def _options(system_prompt: str, model: str, auth: str) -> ClaudeAgentOptions:
    env = None
    if auth == "subscription":
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    return ClaudeAgentOptions(system_prompt=system_prompt, model=model,
                              allowed_tools=[], permission_mode="dontAsk",
                              **({"env": env} if env is not None else {}))


async def _one_shot(prompt: str, system_prompt: str, model: str, auth: str) -> str:
    parts: list[str] = []
    async for message in query(prompt=prompt, options=_options(system_prompt, model, auth)):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                t = getattr(block, "text", None)
                if t:
                    parts.append(t)
    return "".join(parts).strip()


async def generate_query_variations(question: str, n: int,
                                    model: str = "claude-opus-4-8",
                                    auth: str = "auto") -> list[str]:
    prompt = f"Generate {n} alternative phrasings of this query:\n{question}"
    raw = await _one_shot(prompt, _SYS, model, auth)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    variations = lines[:n]
    return [question] + [v for v in variations if v != question]


async def generate_hyde_document(question: str, model: str = "claude-opus-4-8",
                                 auth: str = "auto") -> str:
    return await _one_shot(f"Question: {question}", _HYDE_SYS, model, auth)


async def generate_step_back_question(question: str, model: str = "claude-opus-4-8",
                                      auth: str = "auto") -> str:
    return await _one_shot(f"Question: {question}", _STEPBACK_SYS, model, auth)


_DECOMPOSE_SYS = ("Break the user's question into a few focused sub-questions whose answers "
                  "together answer the original. Output ONLY the sub-questions, one per line.")


async def decompose_question(question: str, n: int, model: str = "claude-opus-4-8",
                             auth: str = "auto") -> list[str]:
    text = await _one_shot(f"Question: {question}\nGive at most {n} sub-questions.",
                           _DECOMPOSE_SYS, model, auth)
    lines = [ln.strip(" -*\t") for ln in text.splitlines() if ln.strip()]
    return lines[:n]
