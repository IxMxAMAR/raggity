from __future__ import annotations

from .llm import LLMProvider
from .models import Chunk

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


# --- CRAG-style corrective retrieval (opt-in) ------------------------------

_CRAG_SYS = (
    "You are a retrieval evaluator. Given a question and the retrieved passages, "
    "grade whether the passages contain enough information to answer the question. "
    "Reply with EXACTLY ONE word and nothing else: correct, incorrect, or ambiguous. "
    "'correct' = the passages clearly contain the answer; "
    "'incorrect' = the passages are irrelevant or lack the answer; "
    "'ambiguous' = partially relevant but insufficient to fully answer.")

_REWRITE_SYS = ("Rewrite this search query to find the missing information. "
                "Reply with ONLY the rewritten query, no commentary, no quotes.")


async def evaluate_retrieval(question: str, chunks: list[Chunk],
                             provider: LLMProvider) -> str:
    """Grade the retrieved *chunks* for *question* with one provider call.

    Returns one of ``"correct" | "incorrect" | "ambiguous"``. Parsing is lenient
    (substring match, ``"incorrect"`` checked before ``"correct"`` since it
    contains it); anything unrecognised maps to ``"ambiguous"``.
    """
    passages = "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(chunks))
    prompt = (f"Question: {question}\n\nRetrieved passages:\n{passages}\n\n"
              "Verdict (one word):")
    raw = (await provider.complete(_CRAG_SYS, prompt)).strip().lower()
    if "incorrect" in raw:
        return "incorrect"
    if "ambiguous" in raw:
        return "ambiguous"
    if "correct" in raw:
        return "correct"
    return "ambiguous"


async def rewrite_query(question: str, provider: LLMProvider) -> str:
    """Rewrite *question* into a search query aimed at the missing information."""
    raw = await provider.complete(_REWRITE_SYS, f"Query: {question}")
    return raw.strip()
