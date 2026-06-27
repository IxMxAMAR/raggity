from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from .embedder import Embedder

log = logging.getLogger("raggity.cached_embedder")


class CachedEmbedder(Embedder):
    def __init__(self, inner: Embedder, cache_path: str) -> None:
        self._inner = inner
        self._path = cache_path
        self._cache = self._load()

    def _load(self) -> dict:
        if not os.path.isfile(self._path):
            return {}
        try:
            with open(self._path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            log.warning("ignoring unreadable embed cache %s: %s", self._path, exc)
            return {}

    def _save(self) -> None:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(self._cache, fh)

    @property
    def dim(self) -> int:
        return self._inner.dim

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing:
            vecs = self._inner.embed_documents([texts[i] for i in missing])
            for i, v in zip(missing, vecs):
                self._cache[keys[i]] = v
            self._save()
        return [self._cache[k] for k in keys]
