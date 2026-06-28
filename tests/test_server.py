import pytest
from fastapi.testclient import TestClient
import raggity.llm as llm_mod
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
        monkeypatch.setattr(llm_mod, "query", _fake_query)
        monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)
        resp = client.post("/ask", json={"question": "how are backups done?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "NAS" in body["answer"]
        assert body["abstained"] is False
        assert isinstance(body["citations"], list)
        for c in body["citations"]:
            assert set(c.keys()) == {"chunk_id", "source_path", "title", "supported"}


# ---------------------------------------------------------------------------
# RED: sessions + SSE (Task 2)
# ---------------------------------------------------------------------------

def test_session_multiturn(tmp_path, monkeypatch):
    """POST /ask with same session_id threads history; response echoes session_id."""
    from raggity.server import create_app
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        client.post("/ingest")
        async def _fake_query(prompt, options):
            yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
        monkeypatch.setattr(llm_mod, "query", _fake_query)
        monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

        r1 = client.post("/ask", json={"question": "how are backups done?", "session_id": "s1"})
        assert r1.status_code == 200
        assert r1.json()["session_id"] == "s1"

        r2 = client.post("/ask", json={"question": "what about restores?", "session_id": "s1"})
        assert r2.status_code == 200
        assert r2.json()["session_id"] == "s1"


def test_ask_no_session_unchanged(tmp_path, monkeypatch):
    """POST /ask without session_id behaves exactly as before (no session_id in response)."""
    from raggity.server import create_app
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        client.post("/ingest")
        async def _fake_query(prompt, options):
            yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
        monkeypatch.setattr(llm_mod, "query", _fake_query)
        monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

        resp = client.post("/ask", json={"question": "how are backups done?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "NAS" in body["answer"]
        assert "session_id" not in body


def test_chat_endpoint_creates_session(tmp_path, monkeypatch):
    """POST /chat always uses a session; returns a session_id even without one provided."""
    from raggity.server import create_app
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        client.post("/ingest")
        async def _fake_query(prompt, options):
            yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
        monkeypatch.setattr(llm_mod, "query", _fake_query)
        monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

        resp = client.post("/chat", json={"question": "how are backups done?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "session_id" in body
        assert len(body["session_id"]) == 32  # uuid4().hex


def test_chat_endpoint_reuses_session(tmp_path, monkeypatch):
    """POST /chat with same session_id reuses the conversation."""
    from raggity.server import create_app
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        client.post("/ingest")
        async def _fake_query(prompt, options):
            yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
        monkeypatch.setattr(llm_mod, "query", _fake_query)
        monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

        r1 = client.post("/chat", json={"question": "how are backups done?", "session_id": "abc"})
        assert r1.json()["session_id"] == "abc"

        r2 = client.post("/chat", json={"question": "what about restores?", "session_id": "abc"})
        assert r2.json()["session_id"] == "abc"


def test_delete_session(tmp_path, monkeypatch):
    """DELETE /session/{id} removes the session; subsequent use creates a new one."""
    from raggity.server import create_app
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        client.post("/ingest")
        async def _fake_query(prompt, options):
            yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
        monkeypatch.setattr(llm_mod, "query", _fake_query)
        monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

        client.post("/chat", json={"question": "how are backups done?", "session_id": "myid"})
        del_resp = client.delete("/session/myid")
        assert del_resp.status_code == 204

        # deleting again returns 404
        del_resp2 = client.delete("/session/myid")
        assert del_resp2.status_code == 404


def test_sse_stream(tmp_path, monkeypatch):
    """GET /ask/stream returns SSE with deltas and a final done event containing citations."""
    from raggity.server import create_app
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        client.post("/ingest")

        async def _fake_query(prompt, options):
            yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
        monkeypatch.setattr(llm_mod, "query", _fake_query)
        monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

        with client.stream("GET", "/ask/stream",
                           params={"question": "how are backups done?"}) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            body = "".join(resp.iter_text())

        assert "NAS" in body
        assert "event: done" in body
        assert "citations" in body


def test_sse_stream_with_session(tmp_path, monkeypatch):
    """GET /ask/stream?session_id=... returns SSE and includes session_id in done event."""
    from raggity.server import create_app
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        client.post("/ingest")

        async def _fake_query(prompt, options):
            yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
        monkeypatch.setattr(llm_mod, "query", _fake_query)
        monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

        with client.stream("GET", "/ask/stream",
                           params={"question": "how are backups done?",
                                   "session_id": "sess99"}) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())

        assert "NAS" in body
        assert "sess99" in body


# ---------------------------------------------------------------------------
# RED: web UI root (Task 3)
# ---------------------------------------------------------------------------

def test_root_serves_web_ui(tmp_path):
    """GET / returns 200, text/html, body contains 'raggity' and references /ask/stream."""
    from raggity.server import create_app
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "raggity" in r.text.lower()
        assert "/ask/stream" in r.text


# ---------------------------------------------------------------------------
# RED: Phase E Task 1 — optional API-key auth + bounded sessions
# ---------------------------------------------------------------------------

def _auth_cfg(tmp_path, auth="none", api_keys=None, max_sessions=1000):
    """Build a config with server auth settings."""
    from raggity.config import ServerConfig
    base = _cfg(tmp_path)
    # Build a new config with server section
    from raggity.config import RaggityConfig, SourcesConfig, IndexConfig
    notes = tmp_path / "notes2"
    notes.mkdir(exist_ok=True)
    (notes / "b.md").write_text("# B\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str(notes / "*.md")]),
        index=IndexConfig(path=str(tmp_path / "idx2")),
        server=ServerConfig(
            auth=auth,
            api_keys=api_keys or [],
            max_sessions=max_sessions,
        ),
    )
    return cfg


def test_auth_none_endpoints_open(tmp_path):
    """auth='none' (default) — all data endpoints accessible without any key."""
    from raggity.server import create_app
    cfg = _auth_cfg(tmp_path, auth="none")
    app = create_app(cfg)
    with TestClient(app) as client:
        assert client.post("/ingest").status_code == 200
        assert client.get("/status").status_code == 200
        assert client.get("/").status_code == 200  # UI always open


def test_auth_api_key_rejects_missing_key(tmp_path):
    """auth='api_key' — requests without any key header get HTTP 401."""
    from raggity.server import create_app
    cfg = _auth_cfg(tmp_path, auth="api_key", api_keys=["secret123"])
    app = create_app(cfg)
    with TestClient(app) as client:
        assert client.post("/ingest").status_code == 401
        assert client.get("/status").status_code == 401
        assert client.post("/ask", json={"question": "q"}).status_code == 401
        assert client.post("/chat", json={"question": "q"}).status_code == 401


def test_auth_api_key_rejects_wrong_key(tmp_path):
    """auth='api_key' — wrong key value yields 401."""
    from raggity.server import create_app
    cfg = _auth_cfg(tmp_path, auth="api_key", api_keys=["secret123"])
    app = create_app(cfg)
    with TestClient(app) as client:
        assert client.get("/status", headers={"X-API-Key": "wrongkey"}).status_code == 401
        assert client.get("/status", headers={"Authorization": "Bearer wrongkey"}).status_code == 401


def test_auth_api_key_accepts_x_api_key_header(tmp_path):
    """auth='api_key' — valid key in X-API-Key header is accepted (200)."""
    from raggity.server import create_app
    cfg = _auth_cfg(tmp_path, auth="api_key", api_keys=["secret123"])
    app = create_app(cfg)
    with TestClient(app) as client:
        client.post("/ingest", headers={"X-API-Key": "secret123"})
        resp = client.get("/status", headers={"X-API-Key": "secret123"})
        assert resp.status_code == 200


def test_auth_api_key_accepts_bearer_token(tmp_path):
    """auth='api_key' — valid key as Bearer token is accepted (200)."""
    from raggity.server import create_app
    cfg = _auth_cfg(tmp_path, auth="api_key", api_keys=["secret123"])
    app = create_app(cfg)
    with TestClient(app) as client:
        client.post("/ingest", headers={"Authorization": "Bearer secret123"})
        resp = client.get("/status", headers={"Authorization": "Bearer secret123"})
        assert resp.status_code == 200


def test_auth_root_always_open(tmp_path):
    """auth='api_key' — GET / (UI) is never gated by auth."""
    from raggity.server import create_app
    cfg = _auth_cfg(tmp_path, auth="api_key", api_keys=["secret123"])
    app = create_app(cfg)
    with TestClient(app) as client:
        # No key provided — UI must still load
        assert client.get("/").status_code == 200


def test_auth_env_keys_merged(tmp_path, monkeypatch):
    """Keys from RAGGITY_API_KEYS env var are merged with config api_keys."""
    from raggity.server import create_app
    monkeypatch.setenv("RAGGITY_API_KEYS", "envkey1,envkey2")
    cfg = _auth_cfg(tmp_path, auth="api_key", api_keys=["cfgkey"])
    app = create_app(cfg)
    with TestClient(app) as client:
        # env key accepted
        assert client.get("/status", headers={"X-API-Key": "envkey1"}).status_code == 200
        assert client.get("/status", headers={"X-API-Key": "envkey2"}).status_code == 200
        # config key still accepted
        assert client.get("/status", headers={"X-API-Key": "cfgkey"}).status_code == 200
        # unknown key rejected
        assert client.get("/status", headers={"X-API-Key": "unknown"}).status_code == 401


def test_session_bounded_evicts_oldest(tmp_path):
    """Sessions dict never exceeds max_sessions; oldest is evicted on overflow."""
    from raggity.server import create_app
    cfg = _auth_cfg(tmp_path, auth="none", max_sessions=3)
    app = create_app(cfg)
    with TestClient(app) as client:
        client.post("/ingest")
        # Create 3 sessions — fills to capacity
        for i in range(3):
            client.post("/chat", json={"question": "q", "session_id": f"s{i}"})
        # Now create a 4th — should evict "s0"
        client.post("/chat", json={"question": "q", "session_id": "s3"})
        # s0 should have been evicted (DELETE returns 404)
        assert client.delete("/session/s0").status_code == 404
        # s1, s2, s3 should still exist
        assert client.delete("/session/s1").status_code == 204
        assert client.delete("/session/s2").status_code == 204
        assert client.delete("/session/s3").status_code == 204


def test_session_lru_access_prevents_eviction(tmp_path):
    """Accessing a session moves it to the end so it is not the next evicted."""
    from raggity.server import create_app
    cfg = _auth_cfg(tmp_path, auth="none", max_sessions=3)
    app = create_app(cfg)
    with TestClient(app) as client:
        client.post("/ingest")
        # Create 3 sessions
        for i in range(3):
            client.post("/chat", json={"question": "q", "session_id": f"s{i}"})
        # Access s0 to move it to the end (most recently used)
        client.post("/chat", json={"question": "q", "session_id": "s0"})
        # Insert s3 — should evict s1 (now the oldest), not s0
        client.post("/chat", json={"question": "q", "session_id": "s3"})
        # s1 should be gone
        assert client.delete("/session/s1").status_code == 404
        # s0 and s2 and s3 should still exist
        assert client.delete("/session/s0").status_code == 204
        assert client.delete("/session/s2").status_code == 204
        assert client.delete("/session/s3").status_code == 204
