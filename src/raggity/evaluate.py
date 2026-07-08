from __future__ import annotations

import json
from dataclasses import dataclass

from .llm import LLMProvider


@dataclass
class EvalResult:
    hit_rate: float
    mrr: float
    recall: float
    n: int  # answerable-row count (rows with answerable != false); denominator above
    # Count of rows marked `"answerable": false` — excluded from hit_rate/mrr/recall
    # (they have no relevant docs to retrieve; see evaluate.llm_judge for their
    # hallucination-resistance metrics: rejection_rate / false_answer_rate).
    unanswerable_total: int = 0


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
    """Raise ValueError with row index if a required key is missing.

    ``"answerable"`` is optional (default ``True`` when absent) and, if
    present, must be a bool: ``false`` marks a row as intentionally
    unanswerable (no relevant docs exist), used to measure hallucination
    resistance — see :func:`evaluate` and :func:`llm_judge`.
    """
    missing = _REQUIRED_KEYS - row.keys()
    if missing:
        raise ValueError(
            f"Golden row {idx} is missing required field(s): {sorted(missing)}. "
            f"Got keys: {sorted(row.keys())}"
        )
    if "answerable" in row and not isinstance(row["answerable"], bool):
        raise ValueError(
            f"Golden row {idx}: 'answerable' must be a bool, got {row['answerable']!r}"
        )


def evaluate(retriever, golden: list[dict], k: int = 5) -> EvalResult:
    """Retrieval-quality metrics (Hit@k, MRR, Recall@k), answerable rows only.

    Rows with ``"answerable": false`` (unanswerable-by-design goldens; see
    docs on hallucination-resistance testing) have no ``relevant_source_paths``
    to score against, so they are skipped from these retrieval metrics and
    counted separately in ``unanswerable_total``. Use :func:`llm_judge` to
    measure whether the system actually abstains on them
    (``rejection_rate`` / ``false_answer_rate``).
    """
    hits = 0.0
    mrr_total = 0.0
    recall_total = 0.0
    answerable_n = 0
    unanswerable_total = 0
    for idx, row in enumerate(golden):
        _validate_row(row, idx)
        if not row.get("answerable", True):
            unanswerable_total += 1
            continue
        answerable_n += 1
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

    n = answerable_n
    if n == 0:
        return EvalResult(0.0, 0.0, 0.0, 0, unanswerable_total)
    return EvalResult(hits / n, mrr_total / n, recall_total / n, n, unanswerable_total)


@dataclass
class JudgeResult:
    faithfulness: float
    answer_relevance: float
    n: int  # answerable-row count; faithfulness/answer_relevance denominator
    # Hallucination-resistance metrics over rows marked `"answerable": false`
    # (CRAG/RGB-style rejection testing). None when the golden set has no
    # unanswerable rows (old two-metric output stays byte-identical).
    rejection_rate: float | None = None  # correct abstentions / unanswerable count
    false_answer_rate: float | None = None  # 1 - rejection_rate; answered anyway


_JUDGE_SYS = (
    "You are a strict grader. Given a QUESTION, CONTEXT, and ANSWER, output EXACTLY "
    "two lines and nothing else:\n"
    "FAITHFUL: YES|NO   (YES if every claim in the ANSWER is supported by the CONTEXT)\n"
    "RELEVANT: YES|NO   (YES if the ANSWER directly addresses the QUESTION)"
)


def _parse_judge(text: str) -> tuple[bool, bool]:
    """Parse the two-labeled-line judge reply leniently.

    A line is truthy only when its label is present AND its value contains
    ``yes``.  Any missing or garbled line defaults to ``False`` — matching the
    previous single-call ``_yes_no`` semantics (absence → NO).
    """
    faith = False
    rel = False
    for line in text.splitlines():
        s = line.strip().lower()
        if s.startswith("faithful:"):
            faith = "yes" in s.split(":", 1)[1]
        elif s.startswith("relevant:"):
            rel = "yes" in s.split(":", 1)[1]
    return faith, rel


async def llm_judge(rag, golden: list[dict], provider: LLMProvider) -> JudgeResult:
    """LLM-judge eval: faithfulness/answer_relevance (answerable rows) plus
    rejection_rate/false_answer_rate (unanswerable rows, hallucination
    resistance). Rows marked ``"answerable": false`` skip the judge call
    entirely — the only signal that matters for them is whether the system
    abstained (correct) or produced any answer (a hallucinated/false answer).
    """
    f_total = 0.0
    r_total = 0.0
    n = 0
    unanswerable_total = 0
    rejection_correct = 0
    for idx, row in enumerate(golden):
        _validate_row(row, idx)
        q = row["question"]
        chunks = rag.retriever.retrieve(q)
        answer = await rag.answerer.answer(q, chunks)
        if not row.get("answerable", True):
            unanswerable_total += 1
            if answer.abstained:
                rejection_correct += 1
            continue
        n += 1
        if answer.abstained or not chunks:
            # abstention is faithful (no unsupported claims) but not relevant
            f_total += 1.0
            continue
        context = "\n\n".join(c.text for c in chunks)
        # ONE provider call per non-abstained row (faithfulness + relevance merged).
        raw = await provider.complete(
            _JUDGE_SYS,
            f"QUESTION:\n{q}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer.text}")
        faith, rel = _parse_judge(raw)
        f_total += 1.0 if faith else 0.0
        r_total += 1.0 if rel else 0.0

    rejection_rate: float | None = None
    false_answer_rate: float | None = None
    if unanswerable_total > 0:
        rejection_rate = rejection_correct / unanswerable_total
        false_answer_rate = 1.0 - rejection_rate

    if n == 0:
        return JudgeResult(0.0, 0.0, 0, rejection_rate, false_answer_rate)
    return JudgeResult(f_total / n, r_total / n, n, rejection_rate, false_answer_rate)
