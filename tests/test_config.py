import pytest
from pydantic import ValidationError

from raggity.config import RaggityConfig, load_config


def test_defaults_when_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.embedding.model == "BAAI/bge-small-en-v1.5"
    assert cfg.embedding.provider == "cpu"
    assert cfg.retrieval.hybrid is True
    assert cfg.retrieval.rrf_k == 60
    assert cfg.retrieval.rerank is True
    assert cfg.retrieval.top_k == 5
    assert cfg.generation.auth == "auto"
    assert cfg.generation.model == "claude-opus-4-8"


def test_explicit_toml_overrides(tmp_path):
    p = tmp_path / "raggity.toml"
    p.write_text(
        '[sources]\ninclude = ["~/notes/**/*.md"]\n'
        '[retrieval]\ntop_k = 8\nrerank = false\n'
        '[generation]\nauth = "subscription"\n'
    )
    cfg = load_config(str(p))
    assert cfg.sources.include == ["~/notes/**/*.md"]
    assert cfg.retrieval.top_k == 8
    assert cfg.retrieval.rerank is False
    assert cfg.generation.auth == "subscription"


def test_local_toml_autodiscovered(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "raggity.toml").write_text('[retrieval]\ntop_k = 3\n')
    cfg = load_config()
    assert cfg.retrieval.top_k == 3


def test_phase2_retrieval_defaults():
    from raggity.config import RaggityConfig
    r = RaggityConfig().retrieval
    assert r.parent_document is False
    assert r.parent_target_tokens == 1024
    assert r.child_target_tokens == 256
    assert r.expand is False
    assert r.expand_n == 3


def test_remaining_defaults():
    from raggity.config import RaggityConfig
    c = RaggityConfig()
    assert c.retrieval.candidates == 30
    assert c.retrieval.dedup_cosine == 0.92
    assert c.retrieval.rerank_model == "Xenova/ms-marco-MiniLM-L-6-v2"
    assert c.index.path == ".raggity/index"
    assert c.sources.include == []


def test_sufficiency_floor_default():
    """Dense-cosine sufficiency floor should default to 0.5."""
    from raggity.config import RetrievalConfig
    r = RetrievalConfig()
    assert r.sufficiency_floor == 0.5


def test_embedding_parallel_defaults_to_none():
    """parallel default is None (in-process single model), NOT 0 (all-cores MP)."""
    from raggity.config import RaggityConfig
    assert RaggityConfig().embedding.parallel is None


def test_sources_exclude_defaults_empty():
    from raggity.config import RaggityConfig
    assert RaggityConfig().sources.exclude == []


def test_relevance_floor_default_is_zero():
    """relevance_floor should default to 0.0 (off) so cross-encoder score never abstains by default."""
    from raggity.config import RetrievalConfig
    r = RetrievalConfig()
    assert r.relevance_floor == 0.0


# ---------------------------------------------------------------------------
# profile = "low-ram"
# ---------------------------------------------------------------------------

def test_profile_defaults_to_empty():
    assert RaggityConfig().profile == ""


def test_profile_empty_is_byte_identical_to_defaults():
    """profile="" must not touch any other field (defaults path unchanged)."""
    plain = RaggityConfig()
    explicit = RaggityConfig(profile="")
    assert explicit.model_dump() == plain.model_dump()


def test_profile_low_ram_forces_expected_fields():
    cfg = RaggityConfig(profile="low-ram")
    assert cfg.index.backend == "lancedb"
    assert cfg.embedding.model == "BAAI/bge-small-en-v1.5"
    assert cfg.embedding.cache is False
    assert cfg.retrieval.rerank is False
    assert cfg.retrieval.graph is False
    assert cfg.generation.cache is False
    assert cfg.server.max_sessions == 100
    assert cfg.server.max_user_rags == 4


def test_profile_low_ram_overrides_conflicting_explicit_values():
    """profile wins over per-field values set in the SAME config, even when they conflict."""
    data = {
        "profile": "low-ram",
        "index": {"backend": "qdrant"},
        "embedding": {"model": "some/other-model", "cache": True},
        "retrieval": {"rerank": True, "graph": True},
        "generation": {"cache": True},
        "server": {"max_sessions": 5000, "max_user_rags": 999},
    }
    cfg = RaggityConfig.model_validate(data)
    assert cfg.index.backend == "lancedb"
    assert cfg.embedding.model == "BAAI/bge-small-en-v1.5"
    assert cfg.embedding.cache is False
    assert cfg.retrieval.rerank is False
    assert cfg.retrieval.graph is False
    assert cfg.generation.cache is False
    assert cfg.server.max_sessions == 100
    assert cfg.server.max_user_rags == 4


def test_profile_invalid_raises_validation_error_naming_choices():
    with pytest.raises(ValidationError) as excinfo:
        RaggityConfig(profile="ultra-fast")
    msg = str(excinfo.value)
    assert "ultra-fast" in msg
    assert "low-ram" in msg


def test_profile_loaded_from_toml(tmp_path):
    p = tmp_path / "raggity.toml"
    p.write_text('profile = "low-ram"\n')
    cfg = load_config(str(p))
    assert cfg.profile == "low-ram"
    assert cfg.retrieval.rerank is False


# ---------------------------------------------------------------------------
# retrieval.rerank_backend (T5: ColBERT late-interaction reranker option)
# ---------------------------------------------------------------------------

def test_rerank_backend_defaults_to_cross_encoder():
    from raggity.config import RetrievalConfig
    r = RetrievalConfig()
    assert r.rerank_backend == "cross-encoder"


def test_rerank_backend_default_is_byte_identical_to_pre_existing_defaults():
    """Adding rerank_backend/colbert_model must not perturb any other default."""
    from raggity.config import RetrievalConfig
    r = RetrievalConfig()
    assert r.rerank is True
    assert r.rerank_model == "Xenova/ms-marco-MiniLM-L-6-v2"


def test_colbert_model_default():
    from raggity.config import RetrievalConfig
    r = RetrievalConfig()
    assert r.colbert_model == "answerdotai/answerai-colbert-small-v1"


def test_rerank_backend_colbert_accepted():
    from raggity.config import RetrievalConfig
    r = RetrievalConfig(rerank_backend="colbert")
    assert r.rerank_backend == "colbert"


def test_rerank_backend_invalid_raises_validation_error_naming_choices():
    from raggity.config import RetrievalConfig
    with pytest.raises(ValidationError) as excinfo:
        RetrievalConfig(rerank_backend="bm25")
    msg = str(excinfo.value)
    assert "bm25" in msg
    assert "cross-encoder" in msg
    assert "colbert" in msg


def test_rerank_backend_loaded_from_toml(tmp_path):
    p = tmp_path / "raggity.toml"
    p.write_text('[retrieval]\nrerank_backend = "colbert"\ncolbert_model = "some/other-colbert"\n')
    cfg = load_config(str(p))
    assert cfg.retrieval.rerank_backend == "colbert"
    assert cfg.retrieval.colbert_model == "some/other-colbert"
