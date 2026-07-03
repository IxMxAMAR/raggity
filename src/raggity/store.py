from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable

from .models import Chunk
from .registry import register

log = logging.getLogger("raggity.store")

TABLE = "chunks"


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: list[Chunk], embedder) -> None: ...
    @abstractmethod
    def delete_source(self, source_path: str) -> None: ...
    def delete_sources(self, source_paths: list[str]) -> None:
        """Delete multiple sources. Default: loop over delete_source (back-compat
        for third-party VectorStore implementations); stores with a native
        batch-delete path should override for a single round-trip."""
        for p in source_paths:
            self.delete_source(p)
    @abstractmethod
    def vector_search(self, query_vec: list[float], limit: int) -> list[Chunk]: ...
    @abstractmethod
    def text_search(self, query: str, limit: int) -> list[Chunk]: ...
    @abstractmethod
    def all_source_paths(self) -> set[str]: ...
    @abstractmethod
    def count(self) -> int: ...
    @abstractmethod
    def optimize(self) -> None: ...
    @abstractmethod
    def reset(self) -> None: ...
    @abstractmethod
    def ensure_ann_index(self, threshold: int) -> None: ...
    @abstractmethod
    def get_by_chunk_ids(self, ids: list[str]) -> list[Chunk]: ...
    @abstractmethod
    def all_chunks(self) -> list[Chunk]: ...
    @classmethod
    @abstractmethod
    def from_config(cls, cfg, dim: "int | Callable[[], int]") -> "VectorStore": ...


def _list_table_names(db) -> list[str]:
    """Return table names from a LanceDB connection.

    Handles both the legacy API (returns an object with a ``.tables``
    attribute) and the modern API (returns a plain ``list[str]``).
    """
    result = db.list_tables()
    if isinstance(result, list):
        return result
    # Legacy object with .tables attribute
    return list(result.tables)


def _row_to_chunk(row: dict, score: float = 0.0) -> Chunk:
    return Chunk(
        text=row["text"],
        source_path=row["source_path"],
        title=row["title"],
        heading_path=row["heading_path"],
        ordinal=int(row["ordinal"]),
        chunk_id=row["chunk_id"],
        score=score,
        parent_id=row.get("parent_id", "") or "",
        parent_text=row.get("parent_text", "") or "",
        vector=row.get("vector"),
    )


class LanceDBStore(VectorStore):
    def __init__(self, path: str, dim: "int | Callable[[], int]") -> None:
        import lancedb

        # ``dim`` may be an int or a zero-arg callable resolved only when a
        # table actually has to be created (deferred so opening an existing
        # index never builds the embedder just to learn its dimension).
        self._dim = dim
        self._db = lancedb.connect(path)
        if TABLE in _list_table_names(self._db):
            self._tbl = self._db.open_table(TABLE)
        else:
            self._tbl = None  # created lazily on first write
        self._fts_ready = False

    def _resolve_dim(self) -> int:
        d = self._dim
        return d() if callable(d) else d

    def _build_schema(self):
        import pyarrow as pa
        dim = self._resolve_dim()
        return pa.schema([
            pa.field("chunk_id", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("title", pa.string()),
            pa.field("heading_path", pa.string()),
            pa.field("ordinal", pa.int64()),
            pa.field("text", pa.string()),
            pa.field("parent_id", pa.string()),
            pa.field("parent_text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ])

    def _require_tbl_for_write(self):
        if self._tbl is None:
            self._tbl = self._db.create_table(TABLE, schema=self._build_schema())
        return self._tbl

    def _ensure_fts(self) -> None:
        if not self._fts_ready:
            try:
                self._tbl.create_fts_index("text", use_tantivy=False, replace=True)
            except Exception:
                log.warning(
                    "raggity.store: could not build FTS (BM25) index — "
                    "hybrid search will degrade to dense-only."
                )
            self._fts_ready = True

    def upsert(self, chunks: list[Chunk], embedder) -> None:
        if not chunks:
            return
        tbl = self._require_tbl_for_write()
        vectors = embedder.embed_documents([c.text for c in chunks])
        rows = [{
            "chunk_id": c.chunk_id,
            "source_path": c.source_path,
            "title": c.title,
            "heading_path": c.heading_path,
            "ordinal": c.ordinal,
            "text": c.text,
            "parent_id": c.parent_id,
            "parent_text": c.parent_text,
            "vector": vec,
        } for c, vec in zip(chunks, vectors)]
        (tbl.merge_insert("chunk_id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows))
        self._fts_ready = False

    def delete_source(self, source_path: str) -> None:
        if self._tbl is None:
            return
        safe = source_path.replace("'", "''")
        self._tbl.delete(f"source_path = '{safe}'")
        self._fts_ready = False

    def delete_sources(self, source_paths: list[str]) -> None:
        if not source_paths or self._tbl is None:
            return
        escaped = ", ".join(
            f"'{p.replace(chr(39), chr(39) + chr(39))}'" for p in source_paths
        )
        self._tbl.delete(f"source_path IN ({escaped})")
        self._fts_ready = False

    def vector_search(self, query_vec: list[float], limit: int) -> list[Chunk]:
        if self._tbl is None:
            return []
        rows = (self._tbl.search(query_vec)
                .metric("cosine").limit(limit).to_list())
        out = []
        for r in rows:
            # lancedb returns _distance (cosine distance); similarity = 1 - dist
            score = 1.0 - float(r.get("_distance", 0.0))
            out.append(_row_to_chunk(r, score))
        return out

    def text_search(self, query: str, limit: int) -> list[Chunk]:
        if self._tbl is None:
            return []
        self._ensure_fts()
        try:
            rows = (self._tbl.search(query, query_type="fts")
                    .limit(limit).to_list())
        except Exception:
            log.warning("raggity.store: FTS query failed — returning empty result.")
            return []
        out = []
        for r in rows:
            score = float(r.get("_score", r.get("score", 0.0)))
            out.append(_row_to_chunk(r, score))
        return out

    def all_source_paths(self) -> set[str]:
        if self.count() == 0:
            return set()
        try:
            rows = self._tbl.to_lance().to_table(columns=["source_path"]).to_pylist()
            return {r["source_path"] for r in rows}
        except Exception:
            # Fallback for lancedb versions where to_lance() API differs
            rows = self._tbl.search().limit(100_000).to_list()
            return {r["source_path"] for r in rows}

    def count(self) -> int:
        if self._tbl is None:
            return 0
        return self._tbl.count_rows()

    def reset(self) -> None:
        if TABLE in _list_table_names(self._db):
            self._db.drop_table(TABLE)
        self._tbl = self._db.create_table(TABLE, schema=self._build_schema())
        self._fts_ready = False

    def optimize(self) -> None:
        if self._tbl is None:
            return
        try:
            self._tbl.optimize()
        except Exception:
            pass

    @classmethod
    def from_config(cls, cfg, dim: "int | Callable[[], int]") -> "LanceDBStore":
        return cls(path=cfg.index.path, dim=dim)

    def ensure_ann_index(self, threshold: int) -> None:
        if self._tbl is None or threshold <= 0 or self.count() < threshold:
            return
        try:
            self._tbl.create_index(metric="cosine", vector_column_name="vector", replace=True)
        except Exception as exc:
            log.warning("ANN index build skipped: %s", exc)

    def get_by_chunk_ids(self, ids: list[str]) -> list[Chunk]:
        if not ids or self._tbl is None:
            return []
        # Build a SQL-safe IN predicate with single-quote escaping
        escaped = ", ".join(f"'{i.replace(chr(39), chr(39)+chr(39))}'" for i in ids)
        predicate = f"chunk_id IN ({escaped})"
        try:
            rows = self._tbl.search().where(predicate, prefilter=True).limit(len(ids)).to_list()
        except Exception:
            # Fallback: bounded page scan — cap at a reasonable max to avoid
            # memory exhaustion on large tables when the primary path fails.
            _FALLBACK_CAP = 50_000
            n = self.count()
            if n > _FALLBACK_CAP:
                log.warning(
                    "raggity.store: get_by_chunk_ids primary path failed; fallback scan "
                    "capped at %d rows (%d total) — some results may be missing.",
                    _FALLBACK_CAP, n,
                )
            id_set = set(ids)
            rows = [r for r in self._tbl.search().limit(_FALLBACK_CAP).to_list()
                    if r["chunk_id"] in id_set]
        return [_row_to_chunk(r) for r in rows]

    def all_chunks(self) -> list[Chunk]:
        """Return all chunks in the store (used for graph building).

        The primary path reads directly from Lance without a vector search
        limit.  The fallback also reads all rows — graph building genuinely
        requires the full corpus — but logs a warning if it had to fall back.
        """
        n = self.count()
        if n == 0:
            return []
        try:
            rows = self._tbl.to_lance().to_table(
                columns=["chunk_id", "source_path", "title", "heading_path",
                         "ordinal", "text", "parent_id", "parent_text"]
            ).to_pylist()
        except Exception:
            # Fallback: search() with full-count limit for graph correctness.
            # This is intentional — all_chunks() must return every chunk for
            # the graph builder; we log so the operator knows the primary path
            # failed, but we do NOT cap here (truncating would break the graph).
            log.warning(
                "raggity.store: all_chunks primary path failed; falling back to "
                "search().limit(%d) — upgrade lancedb if this persists.", n,
            )
            rows = self._tbl.search().limit(n).to_list()
        return [_row_to_chunk(r) for r in rows]


register("store", "lancedb", "raggity.store:LanceDBStore")
register("store", "qdrant", "raggity.qdrant_store:QdrantStore")
