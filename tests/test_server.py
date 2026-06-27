import pytest
from fastapi.testclient import TestClient
import raggity.answerer as answerer_mod
from raggity.config import RaggityConfig, SourcesConfig, IndexConfig


class _Block:
    def __init__(self, text): self.text = text
class _AssistantMessage:
    def __init__(self, text): self.content = [_Block(text)]


def _cfg(tmp_path):
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    return RaggityConfig(sources=SourcesConfig(include=[str(notes / "*.md")]),
                         index=IndexConfig(path=str(tmp_path / "idx")))


def test_server_ingest_status_ask(tmp_path, monkeypatch):
    from raggity.server import create_app
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        assert client.post("/ingest").status_code == 200
        st = client.get("/status").json()
        assert st["chunks"] >= 1
        async def _fake_query(prompt, options):
            yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
        monkeypatch.setattr(answerer_mod, "query", _fake_query)
        monkeypatch.setattr(answerer_mod, "AssistantMessage", _AssistantMessage)
        resp = client.post("/ask", json={"question": "how are backups done?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "NAS" in body["answer"]
        assert body["abstained"] is False
        assert isinstance(body["citations"], list)
        for c in body["citations"]:
            assert set(c.keys()) == {"chunk_id", "source_path", "title", "supported"}
