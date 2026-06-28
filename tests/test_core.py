import pytest
import raggity.llm as llm_mod
from raggity.config import RaggityConfig, SourcesConfig, IndexConfig


class _Block:
    def __init__(self, text): self.text = text


class _AssistantMessage:
    def __init__(self, text): self.content = [_Block(text)]


def test_core_ingest_and_status(tmp_path, monkeypatch):
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str(notes / "*.md")]),
        index=IndexConfig(path=str(tmp_path / "idx")),
    )
    from raggity.core import Raggity
    rag = Raggity(cfg)
    report = rag.ingest()
    assert report.added >= 1
    st = rag.status()
    assert st["chunks"] >= 1


def test_core_ask_uses_pipeline(tmp_path, monkeypatch):
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str(notes / "*.md")]),
        index=IndexConfig(path=str(tmp_path / "idx")),
    )

    async def _fake_query(prompt, options):
        yield _AssistantMessage("Backups run nightly to the NAS [doc_1#" +
                                "00000000].")

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    from raggity.core import Raggity
    rag = Raggity(cfg)
    rag.ingest()
    ans = rag.ask("how are backups done?")
    assert "NAS" in ans.text


def test_core_ask_hyde_routes_retrieve_multi(tmp_path, monkeypatch):
    """aask with hyde=True must call retrieve_multi (not retrieve) and return an answer."""
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str(notes / "*.md")]),
        index=IndexConfig(path=str(tmp_path / "idx")),
    )

    # Single mock handles both HyDE generation and final answer (via llm_mod.query)
    async def _fake_query(prompt, options):
        if "Question:" in prompt:
            yield _AssistantMessage("Backups are stored on a NAS device nightly.")
        else:
            yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    from raggity.core import Raggity
    rag = Raggity(cfg)
    rag.ingest()

    retrieve_multi_calls = []
    original_retrieve_multi = rag.retriever.retrieve_multi
    def _spy_multi(queries, question, **kwargs):
        retrieve_multi_calls.append(queries)
        return original_retrieve_multi(queries, question, **kwargs)
    monkeypatch.setattr(rag.retriever, "retrieve_multi", _spy_multi)

    ans = rag.ask("how are backups done?", hyde=True)
    assert "NAS" in ans.text
    assert len(retrieve_multi_calls) == 1, "retrieve_multi must be called when hyde=True"
    assert len(retrieve_multi_calls[0]) >= 2, "queries list must include original + hyde passage"


def test_aask_cache_hit_skips_model(tmp_path, monkeypatch):
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(sources=SourcesConfig(include=[str(notes / "*.md")]),
                        index=IndexConfig(path=str(tmp_path / "idx")))
    cfg.generation.cache = True
    calls = {"n": 0}
    async def _ans(prompt, options):
        calls["n"] += 1
        yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
    monkeypatch.setattr(llm_mod, "query", _ans)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)
    from raggity.core import Raggity
    rag = Raggity(cfg); rag.ingest()
    a1 = rag.ask("how are backups done?")
    a2 = rag.ask("how are backups done?")   # identical → cache hit, no 2nd model call
    assert a1.text == a2.text
    assert calls["n"] == 1


def test_core_qdrant_backend_ingest_ask(tmp_path, monkeypatch):
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(sources=SourcesConfig(include=[str(notes / "*.md")]),
                        index=IndexConfig(path=str(tmp_path / "idx"), backend="qdrant",
                                          qdrant_location=":memory:", qdrant_collection="t"))
    async def _fake_query(prompt, options):
        yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)
    from raggity.core import Raggity
    rag = Raggity(cfg); rag.ingest()
    assert rag.status()["chunks"] >= 1
    ans = rag.ask("how are backups done?")
    assert "NAS" in ans.text


def test_aask_decompose_merges_and_answers(tmp_path, monkeypatch):
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(sources=SourcesConfig(include=[str(notes / "*.md")]),
                        index=IndexConfig(path=str(tmp_path / "idx")))
    # All LLM calls (decompose + answer) go through llm_mod.query
    async def _fake(prompt, options):
        if "sub-questions" in prompt or "Question:" in prompt and "Give at most" in prompt:
            yield _AssistantMessage("how often?\nwhere stored?")
        else:
            yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
    monkeypatch.setattr(llm_mod, "query", _fake)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)
    from raggity.core import Raggity
    rag = Raggity(cfg); rag.ingest()
    import asyncio
    ans = asyncio.run(rag.aask_decompose("how are backups done?"))
    assert "NAS" in ans.text


def test_core_build_graph_requires_graph_true(tmp_path):
    """build_graph raises RuntimeError when cfg.retrieval.graph=False (default)."""
    from raggity.config import RaggityConfig, IndexConfig
    from raggity.core import Raggity
    import asyncio
    cfg = RaggityConfig(index=IndexConfig(path=str(tmp_path / "idx")))
    rag = Raggity(cfg)
    with pytest.raises(RuntimeError, match="graph"):
        asyncio.run(rag.build_graph())


async def test_core_build_graph_saves_graph_json(tmp_path, monkeypatch):
    """build_graph creates graph.json after extraction (mocked LLM)."""
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    from raggity.config import RaggityConfig, SourcesConfig, IndexConfig, RetrievalConfig
    from raggity.core import Raggity
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str(notes / "*.md")]),
        index=IndexConfig(path=str(tmp_path / "idx")),
        retrieval=RetrievalConfig(graph=True),
    )

    class _Block:
        def __init__(self, t): self.text = t
    class _AM:
        def __init__(self, t): self.content = [_Block(t)]

    async def _fake_query(prompt, options):
        yield _AM("E: NAS\nE: Backup System\nR: Backup System | writes to | NAS\n")

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AM)

    rag = Raggity(cfg)
    rag.ingest()
    # graph.json must exist after ingest with graph=true
    import os
    assert os.path.isfile(str(tmp_path / "idx" / "graph.json"))
    # rag._graph must be loaded
    assert rag._graph is not None
    assert rag._graph.count() >= 1


async def test_core_build_graph_standalone(tmp_path, monkeypatch):
    """Explicit rag graph-build path: ingest first (no graph), then build_graph()."""
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    from raggity.config import RaggityConfig, SourcesConfig, IndexConfig, RetrievalConfig
    from raggity.core import Raggity
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str(notes / "*.md")]),
        index=IndexConfig(path=str(tmp_path / "idx")),
        retrieval=RetrievalConfig(graph=True),
    )

    class _Block:
        def __init__(self, t): self.text = t
    class _AM:
        def __init__(self, t): self.content = [_Block(t)]

    async def _fake_query(prompt, options):
        yield _AM("E: NAS\nE: Backups\n")

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AM)

    rag = Raggity(cfg)
    # Manually ingest without graph building by temporarily disabling graph
    from raggity.config import RetrievalConfig as RC
    import raggity.core as core_mod
    # Ingest without triggering graph build: just call indexer directly
    from raggity.indexer import Indexer
    chunk_kwargs = {"parent_document": False, "parent_target_tokens": 1024,
                    "child_target_tokens": 256}
    indexer = Indexer(rag.embedder, rag.store, rag._manifest_path(),
                      fingerprint=rag._fingerprint(), chunk_kwargs=chunk_kwargs,
                      ann_threshold=rag.cfg.index.ann_threshold)
    indexer.ingest(cfg.sources.include)
    assert rag.store.count() >= 1

    await rag.build_graph()
    import os
    assert os.path.isfile(str(tmp_path / "idx" / "graph.json"))
    assert rag._graph is not None and rag._graph.count() >= 1


def test_core_graph_load_on_init(tmp_path, monkeypatch):
    """Raggity loads graph.json on __init__ when cfg.retrieval.graph=True and file exists."""
    from raggity.config import RaggityConfig, IndexConfig, RetrievalConfig
    from raggity.core import Raggity
    from raggity.graph import GraphStore

    # Write a fake graph.json
    idx = tmp_path / "idx"; idx.mkdir()
    g = GraphStore()
    g.add(["NAS"], [], "c1")
    g.save(str(idx / "graph.json"))

    cfg = RaggityConfig(index=IndexConfig(path=str(idx)),
                        retrieval=RetrievalConfig(graph=True))
    rag = Raggity(cfg)
    assert rag._graph is not None
    assert rag._graph.count() == 1


def test_core_chat_two_turns_appends_conversation(tmp_path, monkeypatch):
    """2-turn chat: conversation accumulates 4 turns (user+assistant×2)."""
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str(notes / "*.md")]),
        index=IndexConfig(path=str(tmp_path / "idx")),
    )

    async def _fake_query(prompt, options):
        yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    from raggity.core import Raggity
    from raggity.conversation import Conversation

    rag = Raggity(cfg)
    rag.ingest()
    conv = Conversation()

    ans1 = rag.chat(conv, "how are backups done?")
    assert "NAS" in ans1.text
    assert len(conv.turns) == 2

    ans2 = rag.chat(conv, "where exactly?")
    assert "NAS" in ans2.text
    assert len(conv.turns) == 4
    assert conv.turns[0] == ("user", "how are backups done?")
    assert conv.turns[2] == ("user", "where exactly?")
