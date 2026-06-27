from __future__ import annotations

import logging
from abc import ABC, abstractmethod

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


def _row_to_chunk(row: dict, score: float = 0.0) -> Chunk:
    return Chunk(
        text=row["text"],
        source_path=row["source_path"],
        title=row["title"],
        heading_path=row["heading_path"],
        ordinal=int(row["ordinal"]),
        chunk_id=row["chunk_id"],
        score=score,
    )


class LanceDBStore(VectorStore):
    def __init__(self, path: str, dim: int) -> None:
        import lancedb
        import pyarrow as pa

        self._dim = dim
        self._db = lancedb.connect(path)
        self._schema = pa.schema([
            pa.field("chunk_id", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("title", pa.string()),
            pa.field("heading_path", pa.string()),
            pa.field("ordinal", pa.int64()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ])
        if TABLE in self._db.list_tables().tables:
            self._tbl = self._db.open_table(TABLE)
        else:
            self._tbl = self._db.create_table(TABLE, schema=self._schema)
        self._fts_ready = False

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
        vectors = embedder.embed_documents([c.text for c in chunks])
        rows = [{
            "chunk_id": c.chunk_id,
            "source_path": c.source_path,
            "title": c.title,
            "heading_path": c.heading_path,
            "ordinal": c.ordinal,
            "text": c.text,
            "vector": vec,
        } for c, vec in zip(chunks, vectors)]
        (self._tbl.merge_insert("chunk_id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows))
        self._fts_ready = False

    def delete_source(self, source_path: str) -> None:
        safe = source_path.replace("'", "''")
        self._tbl.delete(f"source_path = '{safe}'")
        self._fts_ready = False

    def vector_search(self, query_vec: list[float], limit: int) -> list[Chunk]:
        rows = (self._tbl.search(query_vec)
                .metric("cosine").limit(limit).to_list())
        out = []
        for r in rows:
            # lancedb returns _distance (cosine distance); similarity = 1 - dist
            score = 1.0 - float(r.get("_distance", 0.0))
            out.append(_row_to_chunk(r, score))
        return out

    def text_search(self, query: str, limit: int) -> list[Chunk]:
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
        return self._tbl.count_rows()

    def optimize(self) -> None:
        try:
            self._tbl.optimize()
        except Exception:
            pass


register("store", "lancedb", "raggity.store:LanceDBStore")
