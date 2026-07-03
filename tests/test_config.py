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
