from __future__ import annotations

import tomllib
from pathlib import Path

import platformdirs
from pydantic import BaseModel, Field


class SourcesConfig(BaseModel):
    include: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    provider: str = "cpu"  # cpu | cuda | directml | rocm
    batch_size: int = 256
    parallel: int = 0
    cache: bool = False


class RetrievalConfig(BaseModel):
    hybrid: bool = True
    rrf_k: int = 60
    candidates: int = 30
    rerank: bool = True
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
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


class GenerationConfig(BaseModel):
    auth: str = "auto"  # auto | subscription | api_key
    model: str = "claude-opus-4-8"
    cache: bool = False
    backend: str = "claude"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float | None = None


class IndexConfig(BaseModel):
    path: str = ".raggity/index"
    backend: str = "lancedb"
    ann_threshold: int = 50000
    qdrant_location: str = ":memory:"
    qdrant_collection: str = "raggity"
    qdrant_api_key: str | None = None


class RaggityConfig(BaseModel):
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)


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
