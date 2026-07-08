from __future__ import annotations

import tomllib
from pathlib import Path

import platformdirs
from pydantic import BaseModel, Field, field_validator, model_validator

VALID_RERANK_BACKENDS: tuple[str, ...] = ("cross-encoder", "colbert")
# Learned-sparse retrieval choices (Qdrant backend only; LanceDB falls back to
# BM25 FTS with a warning).  "bm25" (default) keeps the existing behavior.
VALID_SPARSE: tuple[str, ...] = ("bm25", "splade", "bm42")


class SourcesConfig(BaseModel):
    include: list[str] = Field(default_factory=list)
    # User glob patterns fnmatch'd against each file's posix path; any match skips
    # the file.  Applied on top of the built-in junk-dir pruning in the loader.
    exclude: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    provider: str = "cpu"  # cpu | cuda | directml | rocm
    batch_size: int = 256
    # None = in-process single-model embedding (the stable path).  fastembed
    # treats 0 as "all cores" MULTIPROCESSING — each worker loads its own ONNX
    # model, which multiplied an OOM crash on Windows.  Opt in explicitly if you
    # really want a worker pool.
    parallel: int | None = None
    cache: bool = False


class RetrievalConfig(BaseModel):
    hybrid: bool = True
    rrf_k: int = 60
    candidates: int = 30
    rerank: bool = True
    # "cross-encoder" (default, uses `rerank_model`) | "colbert" (late-interaction
    # MaxSim scoring via fastembed's LateInteractionTextEmbedding, uses `colbert_model`).
    rerank_backend: str = "cross-encoder"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    colbert_model: str = "answerdotai/answerai-colbert-small-v1"
    top_k: int = 5
    dedup_cosine: float = 0.92
    # Dense-cosine sufficiency floor: governs abstention. Reliable signal (~0.6–0.8
    # for relevant, ~0.43–0.47 for off-topic). When max_dense < this value, abstain.
    sufficiency_floor: float = 0.5
    # OPTIONAL secondary rerank-score filter. 0.0 = off (default). Only applied when
    # rerank=True and relevance_floor > 0. Cross-encoder absolute score is unreliable
    # for abstention, so this is off by default.
    relevance_floor: float = 0.0
    parent_document: bool = False
    parent_target_tokens: int = 1024
    child_target_tokens: int = 256
    expand: bool = False
    expand_n: int = 3
    hyde: bool = False
    step_back: bool = False
    graph: bool = False
    graph_hops: int = 1
    graph_concurrency: int = 8
    # CRAG-style corrective retrieval (opt-in). When true, after the first
    # retrieval a lightweight LLM evaluator grades whether the retrieved passages
    # can answer the question; on "incorrect"/"ambiguous" it runs ONE corrective
    # round (query rewrite + re-retrieve + merge/rerank). Off by default.
    # Cost: +1 LLM call per question, +1 more (rewrite) when a round triggers.
    corrective: bool = False
    # Anthropic-style contextual retrieval (opt-in). When true, `rag ingest`
    # makes one LLM call per new/changed chunk to generate a 1-2 sentence
    # document-context, prepended to the chunk's stored+embedded text. Off by
    # default: massively LLM-cost-heavy for large corpora (one call per chunk).
    contextual: bool = False
    # Bounded concurrency for contextual-retrieval LLM calls during ingest.
    ingest_concurrency: int = 8
    # Learned-sparse retrieval (Qdrant backend only). "bm25" (default) keeps the
    # existing MatchText/BM25 behavior. "splade"/"bm42" store fastembed sparse
    # vectors at upsert and query them at retrieval. On backend=lancedb any
    # non-bm25 value is ignored (warning at store build) and falls back to BM25.
    sparse: str = "bm25"

    @field_validator("rerank_backend")
    @classmethod
    def _validate_rerank_backend(cls, v: str) -> str:
        # Accept the named backends OR a dotted import path ("pkg.module:Class")
        # for a user-supplied custom reranker resolved at build time.
        if v in VALID_RERANK_BACKENDS or ":" in v:
            return v
        choices = ", ".join(f'"{b}"' for b in VALID_RERANK_BACKENDS)
        raise ValueError(
            f"invalid rerank_backend {v!r}; valid choices: {choices}, "
            f'or a dotted import path "package.module:ClassName"')

    @field_validator("sparse")
    @classmethod
    def _validate_sparse(cls, v: str) -> str:
        if v not in VALID_SPARSE:
            choices = ", ".join(f'"{b}"' for b in VALID_SPARSE)
            raise ValueError(f"invalid sparse {v!r}; valid choices: {choices}")
        return v


class GenerationConfig(BaseModel):
    auth: str = "auto"  # auto | subscription | api_key
    model: str = "claude-opus-4-8"
    cache: bool = False
    backend: str = "claude"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float | None = None
    # When true, a local backend (e.g. ollama) is auto-started on first use if a
    # runtime binary is found and the server is not already reachable.
    auto_start: bool = True
    # Opt-in personalization (default off = system prompt byte-identical to today).
    # Free-form persona text appended to the system prompt as user context.
    persona: str = ""
    # When true, tells the model the knowledge base belongs to the current user
    # (first-person docs/questions refer to them).  Grounding rules still bind.
    personal_kb: bool = False
    # Rolling conversation-summary memory (see Conversation.maybe_summarize).
    # When a chat's turn count exceeds this, the oldest turns are compressed
    # into a rolling summary via one provider call and dropped from the turn
    # list; the most-recent memory_max_turns//2 turns are kept verbatim.
    # 0 disables summarization entirely (pure fixed-window behavior).
    memory_max_turns: int = 20


class IndexConfig(BaseModel):
    path: str = ".raggity/index"
    backend: str = "lancedb"
    ann_threshold: int = 50000
    qdrant_location: str = ":memory:"
    qdrant_collection: str = "raggity"
    qdrant_api_key: str | None = None


class ServerConfig(BaseModel):
    auth: str = "none"  # "none" | "api_key"
    api_keys: list[str] = Field(default_factory=list)
    max_sessions: int = 1000
    # When True (requires auth="api_key"), each authenticated identity gets its
    # own namespaced index (multi-tenant).  False (default) = single shared index.
    per_user: bool = False
    # Bound on the per-identity Raggity LRU cache (multi-tenant only).  When the
    # cache exceeds this, the least-recently-used Raggity is evicted and closed.
    max_user_rags: int = 128
    # Optional per-tenant persona text, keyed by API key (identity).  Applied to
    # that tenant's generation config before its Raggity is constructed.
    personas: dict[str, str] = Field(default_factory=dict)


class ObservabilityConfig(BaseModel):
    tracing: bool = False
    service_name: str = "raggity"


# Valid values for the top-level `profile` preset. "" = no preset (default
# behavior, every field governed individually).
VALID_PROFILES: tuple[str, ...] = ("", "low-ram")


class RaggityConfig(BaseModel):
    # Named config preset. "low-ram" hard-overrides several fields below (see
    # `_apply_profile`) to minimize resident memory: embedded lancedb, smallest
    # shipped embedding model, no cross-encoder reranker, no graph expansion,
    # no caches, and capped server session/tenant limits. Profile values win
    # over any explicit per-field value set in the same config file.
    profile: str = ""
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    @field_validator("profile")
    @classmethod
    def _validate_profile(cls, v: str) -> str:
        if v not in VALID_PROFILES:
            choices = ", ".join(f'"{p}"' if p else '"" (none)' for p in VALID_PROFILES)
            raise ValueError(f"invalid profile {v!r}; valid choices: {choices}")
        return v

    @model_validator(mode="after")
    def _apply_profile(self) -> "RaggityConfig":
        """Hard-override select fields when a named profile is active.

        Documented semantics: the profile ALWAYS wins, even over conflicting
        values explicitly set for the same field in the same config file.
        """
        if self.profile == "low-ram":
            self.index.backend = "lancedb"
            self.embedding.model = "BAAI/bge-small-en-v1.5"
            self.embedding.cache = False
            self.retrieval.rerank = False
            self.retrieval.graph = False
            self.generation.cache = False
            self.server.max_sessions = 100
            self.server.max_user_rags = 4
        return self


def _find_config_path(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    local = Path.cwd() / "raggity.toml"
    if local.is_file():
        return local
    user = Path(platformdirs.user_config_dir("raggity")) / "raggity.toml"
    if user.is_file():
        return user
    return None


def load_config(path: str | None = None) -> RaggityConfig:
    cfg_path = _find_config_path(path)
    if cfg_path is None:
        return RaggityConfig()
    with open(cfg_path, "rb") as fh:
        data = tomllib.load(fh)
    return RaggityConfig.model_validate(data)
