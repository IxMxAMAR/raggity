from __future__ import annotations

import json
from dataclasses import dataclass

from .llm import LLMProvider


@dataclass
class EvalResult:
    hit_rate: float
    mrr: float
    recall: float
    n: int


def load_golden(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_REQUIRED_KEYS = {"question"}


def _validate_row(row: dict, idx: int) -> None:
    """Raise ValueError with row index if a required key is missing."""
    missing = _REQUIRED_KEYS - row.keys()
    if missing:
        raise ValueError(
            f"Golden row {idx} is missing required field(s): {sorted(missing)}. "
            f"Got keys: {sorted(row.keys())}"
        )


def evaluate(retriever, golden: list[dict], k: int = 5) -> EvalResult:
    hits = 0.0
    mrr_total = 0.0
    recall_total = 0.0
    for idx, row in enumerate(golden):
        _validate_row(row, idx)
        gold = set(row.get("relevant_source_paths", []))
        retrieved = retriever.retrieve(row["question"])[:k]
        paths = [c.source_path for c in retrieved]

        if not gold:
            # correct behavior is to retrieve nothing (abstain)
            if not paths:
                hits += 1
                mrr_total += 1
                recall_total += 1
            continue

        first_rank = None
        for i, p in enumerate(paths):
            if p in gold:
                first_rank = i + 1
                break
        if first_rank is not None:
            hits += 1
            mrr_total += 1.0 / first_rank
        found = gold & set(paths)
        recall_total += len(found) / len(gold)

    n = len(golden)
    if n == 0:
        return EvalResult(0.0, 0.0, 0.0, 0)
    return EvalResult(hits / n, mrr_total / n, recall_total / n, n)


@dataclass
class JudgeResult:
    faithfulness: float
    answer_relevance: float
    n: int


_FAITH_SYS = ("You are a strict grader. Answer ONLY 'YES' or 'NO'. YES if every claim in the "
              "ANSWER is supported by the CONTEXT; NO otherwise.")
_REL_SYS = ("You are a strict grader. Answer ONLY 'YES' or 'NO'. YES if the ANSWER directly "
            "addresses the QUESTION; NO otherwise.")


async def _yes_no(prompt: str, system_prompt: str, provider: LLMProvider) -> bool:
    result = await provider.complete(system_prompt, prompt)
    return "yes" in result.lower()[:5]


async def llm_judge(rag, golden: list[dict], provider: LLMProvider) -> JudgeResult:
    f_total = 0.0
    r_total = 0.0
    n = 0
    for idx, row in enumerate(golden):
        _validate_row(row, idx)
        q = row["question"]
        chunks = rag.retriever.retrieve(q)
        answer = await rag.answerer.answer(q, chunks)
        n += 1
        if answer.abstained or not chunks:
            # abstention is faithful (no unsupported claims) but not relevant
            f_total += 1.0
            continue
        context = "\n\n".join(c.text for c in chunks)
        faith = await _yes_no(f"CONTEXT:\n{context}\n\nANSWER:\n{answer.text}",
                              _FAITH_SYS, provider)
        rel = await _yes_no(f"QUESTION:\n{q}\n\nANSWER:\n{answer.text}",
                            _REL_SYS, provider)
        f_total += 1.0 if faith else 0.0
        r_total += 1.0 if rel else 0.0
    if n == 0:
        return JudgeResult(0.0, 0.0, 0)
    return JudgeResult(f_total / n, r_total / n, n)
