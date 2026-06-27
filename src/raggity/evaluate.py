from __future__ import annotations

import json
from dataclasses import dataclass

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage


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


def evaluate(retriever, golden: list[dict], k: int = 5) -> EvalResult:
    hits = 0.0
    mrr_total = 0.0
    recall_total = 0.0
    for row in golden:
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


def _judge_options(system_prompt: str, model: str, auth: str) -> ClaudeAgentOptions:
    import os
    env = None
    if auth == "subscription":
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    return ClaudeAgentOptions(system_prompt=system_prompt, model=model,
                              allowed_tools=[], permission_mode="dontAsk",
                              **({"env": env} if env is not None else {}))


async def _yes_no(prompt: str, system_prompt: str, model: str, auth: str) -> bool:
    parts: list[str] = []
    async for message in query(prompt=prompt, options=_judge_options(system_prompt, model, auth)):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                t = getattr(block, "text", None)
                if t:
                    parts.append(t)
    return "yes" in "".join(parts).strip().lower()[:5]


_FAITH_SYS = ("You are a strict grader. Answer ONLY 'YES' or 'NO'. YES if every claim in the "
              "ANSWER is supported by the CONTEXT; NO otherwise.")
_REL_SYS = ("You are a strict grader. Answer ONLY 'YES' or 'NO'. YES if the ANSWER directly "
            "addresses the QUESTION; NO otherwise.")


async def llm_judge(rag, golden: list[dict], model: str = "claude-opus-4-8",
                    auth: str = "auto") -> JudgeResult:
    f_total = 0.0
    r_total = 0.0
    n = 0
    for row in golden:
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
                              _FAITH_SYS, model, auth)
        rel = await _yes_no(f"QUESTION:\n{q}\n\nANSWER:\n{answer.text}",
                            _REL_SYS, model, auth)
        f_total += 1.0 if faith else 0.0
        r_total += 1.0 if rel else 0.0
    if n == 0:
        return JudgeResult(0.0, 0.0, 0)
    return JudgeResult(f_total / n, r_total / n, n)
