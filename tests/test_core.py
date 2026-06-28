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
    def _spy_multi(queries, question):
        retrieve_multi_calls.append(queries)
        return original_retrieve_multi(queries, question)
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
