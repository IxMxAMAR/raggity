import pytest
import raggity.answerer as answerer_mod
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

    monkeypatch.setattr(answerer_mod, "query", _fake_query)
    monkeypatch.setattr(answerer_mod, "AssistantMessage", _AssistantMessage)

    from raggity.core import Raggity
    rag = Raggity(cfg)
    rag.ingest()
    ans = rag.ask("how are backups done?")
    assert "NAS" in ans.text
