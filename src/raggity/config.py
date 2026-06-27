from __future__ import annotations

import os
import tomllib
from pathlib import Path

import platformdirs
from pydantic import BaseModel, Field


class SourcesConfig(BaseModel):
    include: list[str] = Field(default_factory=list)


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    provider: str = "cpu"  # cpu | cuda | directml | rocm


class RetrievalConfig(BaseModel):
    hybrid: bool = True
    rrf_k: int = 60
    candidates: int = 30
    rerank: bool = True
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5
    dedup_cosine: float = 0.92
    relevance_floor: float = 0.3


class GenerationConfig(BaseModel):
    auth: str = "auto"  # auto | subscription | api_key
    model: str = "claude-opus-4-8"


class IndexConfig(BaseModel):
    path: str = ".raggity/index"


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
