from __future__ import annotations
import hashlib
import json
import logging
import os
from pathlib import Path

from .models import Answer, Citation

log = logging.getLogger("raggity.cache")

_DEFAULT_MAX_ENTRIES = 4096


def cache_key(question: str, chunk_ids: list[str], model: str,
              system_prompt: str = "") -> str:
    """Stable cache key that includes system-prompt so prompt changes invalidate entries."""
    payload = (question + "|" + "|".join(sorted(chunk_ids))
               + "|" + model + "|" + system_prompt)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        log.warning("ignoring unreadable answer cache %s: %s", path, exc)
        return {}


def save(path: str, data: dict, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
    """Persist *data* to *path*, trimming oldest entries when over *max_entries* (FIFO)."""
    if len(data) > max_entries:
        # Keep the newest max_entries by discarding from the front of insertion order
        keys = list(data.keys())
        for k in keys[: len(data) - max_entries]:
            del data[k]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def answer_to_dict(a: Answer) -> dict:
    return {"text": a.text, "abstained": a.abstained,
            "citations": [{"chunk_id": c.chunk_id, "source_path": c.source_path,
                           "title": c.title, "supported": c.supported}
                          for c in a.citations]}


def answer_from_dict(d: dict) -> Answer:
    return Answer(text=d["text"],
                  citations=[Citation(c["chunk_id"], c["source_path"], c["title"], c["supported"])
                             for c in d.get("citations", [])],
                  abstained=d.get("abstained", False))
