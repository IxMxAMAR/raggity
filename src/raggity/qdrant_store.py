from __future__ import annotations
import os
import uuid
import warnings
from typing import Callable

from .models import Chunk
from .store import VectorStore

# fastembed SparseTextEmbedding model ids per learned-sparse choice.
_SPARSE_MODELS: dict[str, str] = {
    "splade": "prithivida/Splade_PP_en_v1",
    "bm42": "Qdrant/bm42-all-minilm-l6-v2-attentions",
}
# bm42 emits raw term weights and relies on an IDF modifier at query time;
# SPLADE values are already learned weights and must NOT be IDF-scaled.
_SPARSE_USES_IDF: frozenset[str] = frozenset({"bm42"})


class QdrantStore(VectorStore):
    def __init__(self, location: str = ":memory:",
                 dim: "int | Callable[[], int]" = 384,
                 collection: str = "raggity", api_key: str | None = None,
                 sparse: str = "bm25", sparse_embedder=None) -> None:
        from qdrant_client import QdrantClient, models
        self._m = models
        # ``dim`` may be int or a zero-arg callable resolved only when the
        # collection is actually created (deferred embedder build).
        self._dim = dim
        self._collection = collection
        # Learned-sparse retrieval config.  "bm25" (default) keeps the MatchText
        # scroll path byte-identical; "splade"/"bm42" store & query named sparse
        # vectors.  ``sparse_embedder`` is an optional pre-built (or fake)
        # fastembed SparseTextEmbedding; otherwise it is lazy-built on first use.
        self._sparse = sparse
        self._sparse_enabled = sparse in _SPARSE_MODELS
        self._sparse_model = sparse_embedder
        self._remote = isinstance(location, str) and location.startswith("http")
        if location == ":memory:":
            self._client = QdrantClient(":memory:")
        elif self._remote:
            self._client = QdrantClient(url=location, api_key=api_key)
        else:
            self._client = QdrantClient(path=location)
        self._ensure_collection()

    @classmethod
    def from_config(cls, cfg, dim: "int | Callable[[], int]") -> "QdrantStore":
        api = os.environ.get("QDRANT_API_KEY") or cfg.index.qdrant_api_key
        return cls(location=cfg.index.qdrant_location, dim=dim,
                   collection=cfg.index.qdrant_collection, api_key=api,
                   sparse=getattr(cfg.retrieval, "sparse", "bm25"))

    def _ensure_sparse_model(self):
        """Lazy-build (and cache on the store) the fastembed sparse embedder.

        Shared as a single instance per store so upsert and query use the same
        model.  Never called on the bm25 path.
        """
        if self._sparse_model is None:
            from fastembed import SparseTextEmbedding
            self._sparse_model = SparseTextEmbedding(
                model_name=_SPARSE_MODELS[self._sparse])
        return self._sparse_model

    def _to_sparse_vector(self, emb):
        """Convert a fastembed SparseEmbedding (indices/values) to a qdrant
        SparseVector, normalizing numpy/list inputs to plain Python scalars."""
        return self._m.SparseVector(
            indices=[int(i) for i in emb.indices],
            values=[float(v) for v in emb.values])

    def _ensure_collection(self) -> None:
        m = self._m
        if not self._client.collection_exists(self._collection):
            dim = self._dim() if callable(self._dim) else self._dim
            extra = {}
            if self._sparse_enabled:
                modifier = (m.Modifier.IDF
                            if self._sparse in _SPARSE_USES_IDF else None)
                extra["sparse_vectors_config"] = {
                    "sparse": m.SparseVectorParams(modifier=modifier)}
            self._client.create_collection(
                self._collection,
                vectors_config=m.VectorParams(size=dim, distance=m.Distance.COSINE),
                **extra,
            )
            # Create payload indexes for all modes (remote, local, :memory:).
            # Best-effort: some older qdrant-client / qdrant-local builds reject index
            # creation on in-memory collections — degrade gracefully if so.
            # NOTE: `text` is indexed as a TEXT (MatchText keyword-match) field, which
            # enables substring/token filtering but is NOT true BM25 ranking (parity gap
            # vs LanceDB's Tantivy-backed FTS).  BM25 scoring requires a remote Qdrant
            # server; local/:memory: falls back to rank-by-position in text_search().
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress local-mode "no effect" UserWarning
                try:
                    self._client.create_payload_index(
                        self._collection, "text",
                        field_schema=m.TextIndexParams(type=m.TextIndexType.TEXT))
                except Exception:
                    pass
                try:
                    self._client.create_payload_index(
                        self._collection, "source_path",
                        field_schema=m.PayloadSchemaType.KEYWORD)
                except Exception:
                    pass

    def _pid(self, chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))

    def _payload(self, c: Chunk) -> dict:
        return {"chunk_id": c.chunk_id, "source_path": c.source_path, "title": c.title,
                "heading_path": c.heading_path, "ordinal": c.ordinal, "text": c.text,
                "parent_id": c.parent_id, "parent_text": c.parent_text}

    def _to_chunk(self, p: object, score: float) -> Chunk:
        payload = p.payload  # type: ignore[attr-defined]
        vec = getattr(p, "vector", None)
        # In sparse mode points carry named vectors, so ``vector`` is returned as
        # a dict {"": <dense list>, "sparse": SparseVector}.  Keep only the dense
        # component (used by dedup); the bm25 path returns a plain list untouched.
        if isinstance(vec, dict):
            vec = vec.get("")
        return Chunk(text=payload["text"], source_path=payload["source_path"],
                     title=payload["title"], heading_path=payload["heading_path"],
                     ordinal=int(payload["ordinal"]), chunk_id=payload["chunk_id"],
                     score=score,
                     parent_id=payload.get("parent_id", "") or "",
                     parent_text=payload.get("parent_text", "") or "",
                     vector=vec)

    def upsert(self, chunks: list[Chunk], embedder) -> None:
        if not chunks:
            return
        m = self._m
        texts = [c.text for c in chunks]
        vectors = embedder.embed_documents(texts)
        if self._sparse_enabled:
            sparse_embs = list(self._ensure_sparse_model().embed(texts))
            points = [
                m.PointStruct(
                    id=self._pid(c.chunk_id),
                    vector={"": v, "sparse": self._to_sparse_vector(se)},
                    payload=self._payload(c))
                for c, v, se in zip(chunks, vectors, sparse_embs)]
        else:
            points = [m.PointStruct(id=self._pid(c.chunk_id), vector=v,
                                    payload=self._payload(c))
                      for c, v in zip(chunks, vectors)]
        self._client.upsert(self._collection, points=points)

    def delete_source(self, source_path: str) -> None:
        m = self._m
        self._client.delete(self._collection, points_selector=m.FilterSelector(
            filter=m.Filter(must=[m.FieldCondition(key="source_path",
                                                   match=m.MatchValue(value=source_path))])))

    def delete_sources(self, source_paths: list[str]) -> None:
        if not source_paths:
            return
        m = self._m
        self._client.delete(self._collection, points_selector=m.FilterSelector(
            filter=m.Filter(must=[m.FieldCondition(key="source_path",
                                                   match=m.MatchAny(any=source_paths))])))

    def vector_search(self, query_vec: list[float], limit: int) -> list[Chunk]:
        res = self._client.query_points(self._collection, query=query_vec,
                                        limit=limit, with_payload=True,
                                        with_vectors=True).points
        return [self._to_chunk(p, float(p.score)) for p in res]

    def text_search(self, query: str, limit: int) -> list[Chunk]:
        m = self._m
        if self._sparse_enabled:
            # Learned-sparse retrieval: embed the query sparsely and rank by the
            # named sparse vector (true sparse scoring, not MatchText position).
            qse = next(iter(self._ensure_sparse_model().query_embed(query)))
            sv = self._to_sparse_vector(qse)
            if not sv.indices:
                return []
            try:
                res = self._client.query_points(
                    self._collection, query=sv, using="sparse", limit=limit,
                    with_payload=True, with_vectors=True).points
            except Exception:
                return []
            return [self._to_chunk(p, float(p.score)) for p in res]
        try:
            res, _ = self._client.scroll(
                self._collection,
                scroll_filter=m.Filter(must=[m.FieldCondition(
                    key="text", match=m.MatchText(text=query))]),
                limit=limit, with_payload=True, with_vectors=True)
        except Exception:
            return []
        return [self._to_chunk(p, 1.0 / (rank + 1)) for rank, p in enumerate(res)]

    def all_source_paths(self) -> set[str]:
        paths: set[str] = set()
        offset = None
        while True:
            res, offset = self._client.scroll(self._collection, limit=256,
                                              with_payload=["source_path"], offset=offset)
            for p in res:
                paths.add(p.payload["source_path"])
            if offset is None:
                break
        return paths

    def count(self) -> int:
        return self._client.count(self._collection).count

    def optimize(self) -> None:
        pass

    def ensure_ann_index(self, threshold: int) -> None:
        pass  # Qdrant builds HNSW automatically

    def reset(self) -> None:
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._ensure_collection()

    def all_chunks(self) -> list[Chunk]:
        """Return all chunks in the store (used for graph building)."""
        out: list[Chunk] = []
        offset = None
        while True:
            res, offset = self._client.scroll(
                self._collection, limit=256, with_payload=True, offset=offset)
            for p in res:
                out.append(self._to_chunk(p, 0.0))
            if offset is None:
                break
        return out

    def get_by_chunk_ids(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        m = self._m
        # Retrieve by deterministic uuid5 point ids
        point_ids = [self._pid(cid) for cid in ids]
        try:
            results = self._client.retrieve(
                self._collection, ids=point_ids, with_payload=True, with_vectors=True
            )
            return [self._to_chunk(p, 0.0) for p in results]
        except Exception:
            # Fallback: scroll with chunk_id filter
            id_set = set(ids)
            out: list[Chunk] = []
            offset = None
            while True:
                res, offset = self._client.scroll(
                    self._collection,
                    scroll_filter=m.Filter(should=[
                        m.FieldCondition(key="chunk_id", match=m.MatchValue(value=cid))
                        for cid in ids
                    ]),
                    limit=len(ids) + 1,
                    with_payload=True,
                    with_vectors=True,
                    offset=offset,
                )
                for p in res:
                    if p.payload.get("chunk_id") in id_set:
                        out.append(self._to_chunk(p, 0.0))
                if offset is None:
                    break
            return out
