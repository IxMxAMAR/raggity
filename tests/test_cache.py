from raggity.models import Answer, Citation
from raggity import cache


def test_cache_key_stable_and_order_independent():
    k1 = cache.cache_key("q", ["b", "a"], "m")
    k2 = cache.cache_key("q", ["a", "b"], "m")
    assert k1 == k2  # chunk id order independent
    assert cache.cache_key("q", ["a"], "m") != cache.cache_key("q2", ["a"], "m")


def test_answer_roundtrip():
    a = Answer(text="x", citations=[Citation("c1", "a.md", "A", True)], abstained=False)
    d = cache.answer_to_dict(a)
    a2 = cache.answer_from_dict(d)
    assert a2.text == "x" and a2.citations[0].chunk_id == "c1" and a2.abstained is False


def test_load_tolerates_missing_and_corrupt(tmp_path):
    p = str(tmp_path / "answer_cache.json")
    assert cache.load(p) == {}
    with open(p, "w") as fh:
        fh.write("{ not json")
    assert cache.load(p) == {}


def test_cache_key_includes_system_prompt():
    """Different system_prompt → different cache key."""
    k1 = cache.cache_key("q", ["a"], "m", system_prompt="prompt-v1")
    k2 = cache.cache_key("q", ["a"], "m", system_prompt="prompt-v2")
    assert k1 != k2


def test_cache_key_default_system_prompt_stable():
    """Omitting system_prompt produces the same key each time (deterministic)."""
    k1 = cache.cache_key("q", ["a"], "m")
    k2 = cache.cache_key("q", ["a"], "m")
    assert k1 == k2


def test_save_trims_to_max_entries(tmp_path):
    """save() evicts oldest entries when data exceeds max_entries."""
    p = str(tmp_path / "cache.json")
    # Build a dict with 5 entries; max_entries=3 → only 3 newest survive
    data = {str(i): {"text": f"v{i}", "abstained": False, "citations": []}
            for i in range(5)}
    cache.save(p, data, max_entries=3)
    saved = cache.load(p)
    assert len(saved) == 3


import asyncio

async def test_concurrent_aask_both_persist(tmp_path, monkeypatch):
    """Two concurrent aask() calls must both persist their results without losing entries."""
    import raggity.llm as llm_mod

    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    from raggity.config import RaggityConfig, SourcesConfig, IndexConfig
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str(notes / "*.md")]),
        index=IndexConfig(path=str(tmp_path / "idx")),
    )
    cfg.generation.cache = True

    class _Block:
        def __init__(self, t): self.text = t
    class _AM:
        def __init__(self, t): self.content = [_Block(t)]

    async def _fake_query(prompt, options):
        yield _AM("answer text [doc_1#00000000].")

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AM)

    from raggity.core import Raggity
    rag = Raggity(cfg)
    rag.ingest()

    # Two different questions → two distinct cache entries
    a1, a2 = await asyncio.gather(
        rag.aask("question one?"),
        rag.aask("question two?"),
    )
    # Both should have produced answers
    assert a1.text and a2.text
    # Both cache entries must be persisted
    import raggity.cache as cache_mod
    data = cache_mod.load(rag._cache_path())
    assert len(data) == 2, f"Expected 2 cache entries, got {len(data)}"


async def test_system_prompt_change_invalidates_cache(tmp_path, monkeypatch):
    """Changing SYSTEM_PROMPT makes aask() produce a cache miss."""
    import raggity.llm as llm_mod
    import raggity.cache as cache_mod
    import raggity.prompts as prompts_mod

    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    from raggity.config import RaggityConfig, SourcesConfig, IndexConfig
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str(notes / "*.md")]),
        index=IndexConfig(path=str(tmp_path / "idx")),
    )
    cfg.generation.cache = True

    calls = {"n": 0}

    class _Block:
        def __init__(self, t): self.text = t
    class _AM:
        def __init__(self, t): self.content = [_Block(t)]

    async def _fake_query(prompt, options):
        calls["n"] += 1
        yield _AM("answer text [doc_1#00000000].")

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AM)

    from raggity.core import Raggity
    rag = Raggity(cfg)
    rag.ingest()

    await rag.aask("how are backups done?")
    assert calls["n"] == 1

    # Simulate system prompt change
    original = prompts_mod.SYSTEM_PROMPT
    monkeypatch.setattr(prompts_mod, "SYSTEM_PROMPT", "NEW SYSTEM PROMPT v2")
    await rag.aask("how are backups done?")
    assert calls["n"] == 2, "Prompt change must produce a cache miss"
