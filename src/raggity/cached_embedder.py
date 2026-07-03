from __future__ import annotations

import array
import hashlib
import json
import logging
import os
import sqlite3
import threading
from pathlib import Path

from .embedder import Embedder

log = logging.getLogger("raggity.cached_embedder")

# SQLite bound variable limit is 999 by default; stay well under it per SELECT IN.
_SQLITE_MAX_VARS = 900


def _pack(vec: list[float]) -> bytes:
    """Serialise a float vector to compact float32 bytes."""
    return array.array("f", vec).tobytes()


def _unpack(blob: bytes) -> list[float]:
    a = array.array("f")
    a.frombytes(blob)
    return a.tolist()


class CachedEmbedder(Embedder):
    """Embedder wrapper backed by a persistent SQLite embed cache.

    The cache stores ``(key TEXT PRIMARY KEY, vec BLOB)`` where the blob is a
    float32-packed vector.  Writes are ``O(new)`` (``executemany INSERT OR
    REPLACE``) rather than rewriting the whole cache each call.  A one-time
    migration imports any legacy ``embed_cache.json`` sitting alongside the DB
    (then renames it ``.json.bak``).  A corrupt/unusable DB degrades to plain
    re-embedding (no caching) rather than raising.
    """

    def __init__(self, inner: Embedder, cache_path: str) -> None:
        self._inner = inner
        self._path = cache_path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._degraded = False  # True once the DB is found unusable

    # -- connection (lazy, double-checked) --------------------------------
    def _connect(self) -> sqlite3.Connection | None:
        if self._conn is not None or self._degraded:
            return self._conn
        with self._lock:
            if self._conn is not None or self._degraded:
                return self._conn
            try:
                Path(self._path).parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(self._path, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS embed_cache "
                    "(key TEXT PRIMARY KEY, vec BLOB)"
                )
                conn.commit()
                self._migrate_json(conn)
                self._conn = conn
            except Exception as exc:
                log.warning(
                    "embed cache %s unusable (%s); re-embedding without cache",
                    self._path, exc,
                )
                self._degraded = True
                self._conn = None
            return self._conn

    def _migrate_json(self, conn: sqlite3.Connection) -> None:
        """Import a legacy JSON embed cache (once), then retire it to .bak."""
        json_path = os.path.splitext(self._path)[0] + ".json"
        if not os.path.isfile(json_path):
            return
        try:
            with open(json_path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            log.warning("could not read legacy embed cache %s: %s", json_path, exc)
            return
        rows = []
        for k, v in data.items():
            try:
                rows.append((k, _pack(v)))
            except Exception:
                continue
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO embed_cache (key, vec) VALUES (?, ?)", rows)
            conn.commit()
        try:
            os.replace(json_path, json_path + ".bak")
        except Exception:
            pass

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    # -- Embedder interface ----------------------------------------------
    @property
    def dim(self) -> int:
        return self._inner.dim

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)

    @property
    def _model_id(self) -> str:
        """Stable model identifier for cache-key scoping.

        Uses ``model_id`` if the inner embedder exposes it (e.g. tests),
        otherwise falls back to the class name so different embedder types
        never share keys.
        """
        return getattr(self._inner, "model_id", type(self._inner).__name__)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model_prefix = f"{self._model_id}|{self._inner.dim}|"
        keys = [
            hashlib.sha256((model_prefix + t).encode("utf-8")).hexdigest()
            for t in texts
        ]
        conn = self._connect()
        if conn is None:  # degraded — no caching
            return self._inner.embed_documents(texts)

        found: dict[str, list[float]] = {}
        with self._lock:
            for i in range(0, len(keys), _SQLITE_MAX_VARS):
                batch = keys[i:i + _SQLITE_MAX_VARS]
                placeholders = ",".join("?" * len(batch))
                cur = conn.execute(
                    f"SELECT key, vec FROM embed_cache WHERE key IN ({placeholders})",
                    batch,
                )
                for k, blob in cur.fetchall():
                    found[k] = _unpack(blob)

        result: list[list[float] | None] = [None] * len(keys)
        missing_idx: list[int] = []
        for i, k in enumerate(keys):
            if k in found:
                result[i] = found[k]
            else:
                missing_idx.append(i)

        if missing_idx:
            vecs = self._inner.embed_documents([texts[i] for i in missing_idx])
            rows = []
            for j, i in enumerate(missing_idx):
                result[i] = vecs[j]
                rows.append((keys[i], _pack(vecs[j])))
            with self._lock:
                conn.executemany(
                    "INSERT OR REPLACE INTO embed_cache (key, vec) VALUES (?, ?)",
                    rows,
                )
                conn.commit()
        return result  # type: ignore[return-value]
