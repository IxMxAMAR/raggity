from __future__ import annotations

import json
from dataclasses import dataclass


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
