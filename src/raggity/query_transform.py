from __future__ import annotations

from .llm import LLMProvider

_SYS = ("You generate alternative phrasings of a search query to improve retrieval. "
        "Output ONLY the alternative queries, one per line, no numbering, no commentary.")

_HYDE_SYS = ("Write a short, plausible answer passage for the user's question, as if "
             "excerpted from a relevant document. Output only the passage, no preamble.")

_STEPBACK_SYS = ("Given a specific question, produce ONE broader, more general question "
                 "that would surface useful background. Output only the question.")

_DECOMPOSE_SYS = ("Break the user's question into a few focused sub-questions whose answers "
                  "together answer the original. Output ONLY the sub-questions, one per line.")


async def generate_query_variations(question: str, n: int,
                                    provider: LLMProvider) -> list[str]:
    prompt = f"Generate {n} alternative phrasings of this query:\n{question}"
    raw = await provider.complete(_SYS, prompt)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    variations = lines[:n]
    return [question] + [v for v in variations if v != question]


async def generate_hyde_document(question: str, provider: LLMProvider) -> str:
    return await provider.complete(_HYDE_SYS, f"Question: {question}")


async def generate_step_back_question(question: str, provider: LLMProvider) -> str:
    return await provider.complete(_STEPBACK_SYS, f"Question: {question}")


async def decompose_question(question: str, n: int, provider: LLMProvider) -> list[str]:
    text = await provider.complete(_DECOMPOSE_SYS,
                                   f"Question: {question}\nGive at most {n} sub-questions.")
    lines = [ln.strip(" -*\t") for ln in text.splitlines() if ln.strip()]
    return lines[:n]
