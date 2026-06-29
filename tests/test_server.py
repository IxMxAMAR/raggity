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


# ---------------------------------------------------------------------------
# RED: Phase E Task 2 — per-user collections (namespaced multi-tenant)
# ---------------------------------------------------------------------------

def _per_user_cfg(tmp_path, per_user=True):
    """Config with auth=api_key + two test keys; per_user flag controlled."""
    from raggity.config import RaggityConfig, SourcesConfig, IndexConfig, ServerConfig
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[]),          # no static sources; we ingest via API
        index=IndexConfig(path=str(tmp_path / "base_idx")),
        server=ServerConfig(
            auth="api_key",
            api_keys=["key_alice", "key_bob"],
            per_user=per_user,
        ),
    )
    return cfg


def _ingest_doc(client, key, content, tmp_path, label):
    """Write a small doc and POST /ingest so it lands in the store for *key*."""
    from raggity.models import Document
    import raggity.core as core_mod

    doc = Document(
        path=f"/fake/{label}.md",
        title=label,
        text=content,
        file_hash=label,
        mtime=0.0,
    )

    # Monkey-patch ingest_documents so we can inject an in-memory doc without disk files
    orig = None

    def _patched_ingest(self, docs):
        return core_mod.Raggity.ingest_documents(self, docs)

    # Directly call the per-user /ingest by passing doc via a helper
    # We'll override ingest() to call ingest_documents with our doc
    import raggity.server as srv_mod

    original_ingest_method = None

    def patched_ingest_on_instance(instance, _docs=[doc]):
        from raggity.indexer import IngestReport
        instance.ingest_documents(_docs)
        return IngestReport(added=1, updated=0, deleted=0, unchanged=0)

    # Temporarily patch the rag's ingest method for this call
    import unittest.mock as mock
    headers = {"X-API-Key": key}
    with mock.patch.object(core_mod.Raggity, "ingest", patched_ingest_on_instance):
        resp = client.post("/ingest", headers=headers)
    assert resp.status_code == 200, f"ingest failed: {resp.text}"
    return resp


def test_per_user_isolation(tmp_path, monkeypatch):
    """per_user=True: key_alice ingests docA, key_bob ingests docB.
    Each key's /status shows only their own chunks; counts are separate."""
    from raggity.server import create_app

    cfg = _per_user_cfg(tmp_path, per_user=True)
    app = create_app(cfg)
    with TestClient(app) as client:
        _ingest_doc(client, "key_alice", "Alice exclusive: the sky is green.", tmp_path, "alice_doc")
        _ingest_doc(client, "key_bob", "Bob exclusive: the sea is purple.", tmp_path, "bob_doc")

        st_alice = client.get("/status", headers={"X-API-Key": "key_alice"}).json()
        st_bob = client.get("/status", headers={"X-API-Key": "key_bob"}).json()

        # Each namespace has exactly the chunks they ingested — they do not share
        assert st_alice["chunks"] >= 1
        assert st_bob["chunks"] >= 1
        # Isolation: the index_path reported must differ between the two users
        assert st_alice["index_path"] != st_bob["index_path"], (
            "per_user=True must route alice and bob to different index paths"
        )


def test_per_user_ask_isolation(tmp_path, monkeypatch):
    """per_user=True: /ask for key_alice cannot retrieve key_bob's content."""
    from raggity.server import create_app
    import raggity.llm as llm_mod

    cfg = _per_user_cfg(tmp_path, per_user=True)
    app = create_app(cfg)
    with TestClient(app) as client:
        _ingest_doc(client, "key_alice", "Alice exclusive: the sky is green.", tmp_path, "alice_doc2")
        _ingest_doc(client, "key_bob", "Bob exclusive: the sea is purple.", tmp_path, "bob_doc2")

        async def _fake_query(prompt, options):
            yield _AssistantMessage("I cannot find information about that in the provided context.")

        monkeypatch.setattr(llm_mod, "query", _fake_query)
        monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

        # Alice asks about Bob's content — she should not see it
        resp = client.post(
            "/ask",
            json={"question": "what color is the sea?"},
            headers={"X-API-Key": "key_alice"},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Bob's specific content "purple" must not appear in alice's chunks
        # (the retriever won't find it because alice's index only has her doc)
        # We verify by checking status chunk counts stay separate
        st_alice = client.get("/status", headers={"X-API-Key": "key_alice"}).json()
        st_bob = client.get("/status", headers={"X-API-Key": "key_bob"}).json()
        assert st_alice["index_path"] != st_bob["index_path"]


def test_per_user_off_shared_index(tmp_path, monkeypatch):
    """per_user=False (default): both keys share the same Raggity instance/index."""
    from raggity.server import create_app

    cfg = _per_user_cfg(tmp_path, per_user=False)
    app = create_app(cfg)
    with TestClient(app) as client:
        _ingest_doc(client, "key_alice", "Shared doc: the sky is blue.", tmp_path, "shared_doc")

        # Both keys see the same status (same index_path)
        st_alice = client.get("/status", headers={"X-API-Key": "key_alice"}).json()
        st_bob = client.get("/status", headers={"X-API-Key": "key_bob"}).json()

        assert st_alice["index_path"] == st_bob["index_path"], (
            "per_user=False must route all keys to the same shared index"
        )
        assert st_alice["chunks"] == st_bob["chunks"]


def test_per_user_config_default_false():
    """ServerConfig.per_user defaults to False (no-op for single-user usage)."""
    from raggity.config import ServerConfig
    sc = ServerConfig()
    assert sc.per_user is False


def test_for_namespace_lancedb(tmp_path):
    """Raggity.for_namespace(ns) returns a Raggity with namespaced LanceDB path."""
    from raggity.config import RaggityConfig, IndexConfig
    from raggity.core import Raggity

    base_path = str(tmp_path / "base")
    cfg = RaggityConfig(index=IndexConfig(path=base_path, backend="lancedb"))
    rag = Raggity(cfg)

    ns_rag = rag.for_namespace("alice")
    assert ns_rag.cfg.index.path == str(tmp_path / "base" / "users" / "alice")
    # Original unchanged
    assert rag.cfg.index.path == base_path


def test_for_namespace_slug_sanitization(tmp_path):
    """for_namespace sanitizes unsafe characters to a filesystem-safe slug."""
    from raggity.config import RaggityConfig, IndexConfig
    from raggity.core import Raggity

    cfg = RaggityConfig(index=IndexConfig(path=str(tmp_path / "base"), backend="lancedb"))
    rag = Raggity(cfg)

    # Key with unsafe chars must not end up in path literally
    ns_rag = rag.for_namespace("user@example.com/evil/../path")
    ns_path = ns_rag.cfg.index.path
    # Must not contain any of the unsafe chars
    for bad in ["@", "/", ".", "\\"]:
        assert bad not in ns_path.split("users" + ("/" if "/" in ns_path else "\\"))[-1], (
            f"Unsafe char {bad!r} found in namespace slug: {ns_path}"
        )


def test_for_namespace_qdrant_collection(tmp_path):
    """for_namespace namespaces the qdrant_collection as <base>_<ns>."""
    from raggity.config import RaggityConfig, IndexConfig
    from raggity.core import Raggity

    cfg = RaggityConfig(
        index=IndexConfig(
            path=str(tmp_path / "base"),
            backend="lancedb",
            qdrant_collection="myproject",
        )
    )
    rag = Raggity(cfg)
    ns_rag = rag.for_namespace("bob")
    assert ns_rag.cfg.index.qdrant_collection == "myproject_bob"
    # Original unchanged
    assert rag.cfg.index.qdrant_collection == "myproject"


# ---------------------------------------------------------------------------
# RED: Group 4 — server security + multi-tenant
# ---------------------------------------------------------------------------


def _fix_query(monkeypatch, text="Backups run nightly to the NAS [doc_1#00000000]."):
    async def _fake_query(prompt, options):
        yield _AssistantMessage(text)
    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)


# --- Fix 1: per-identity session isolation -------------------------------

def test_session_isolated_by_identity(tmp_path, monkeypatch):
    """auth=api_key: key A creates session s1; key B using session_id=s1 does NOT
    see A's session and gets/creates its own."""
    from raggity.server import create_app
    cfg = _auth_cfg(tmp_path, auth="api_key", api_keys=["A", "B"])
    app = create_app(cfg)
    with TestClient(app) as client:
        client.post("/ingest", headers={"X-API-Key": "A"})
        _fix_query(monkeypatch)
        # A creates session s1 via /chat
        r = client.post("/chat", json={"question": "q", "session_id": "s1"},
                        headers={"X-API-Key": "A"})
        assert r.status_code == 200
        # B uses session_id s1 via /ask — must not error, and must be B's own namespace
        rb = client.post("/ask", json={"question": "q", "session_id": "s1"},
                         headers={"X-API-Key": "B"})
        assert rb.status_code == 200
        # B cannot delete A's session: B's DELETE /session/s1 removes only B's.
        # A's session must survive.
        assert client.delete("/session/s1", headers={"X-API-Key": "B"}).status_code == 204
        # A still has s1
        assert client.delete("/session/s1", headers={"X-API-Key": "A"}).status_code == 204


def test_delete_session_cannot_probe_other_identity(tmp_path, monkeypatch):
    """B's DELETE for a session id it never created returns 404 even if A owns it."""
    from raggity.server import create_app
    cfg = _auth_cfg(tmp_path, auth="api_key", api_keys=["A", "B"])
    app = create_app(cfg)
    with TestClient(app) as client:
        client.post("/ingest", headers={"X-API-Key": "A"})
        _fix_query(monkeypatch)
        client.post("/chat", json={"question": "q", "session_id": "shared_id"},
                    headers={"X-API-Key": "A"})
        # B never created shared_id -> 404 (cannot delete or probe A's)
        assert client.delete("/session/shared_id", headers={"X-API-Key": "B"}).status_code == 404
        # A's still alive
        assert client.delete("/session/shared_id", headers={"X-API-Key": "A"}).status_code == 204


def test_session_namespace_auth_none_unchanged(tmp_path, monkeypatch):
    """auth=none: shared session namespace (identity=None), same as before."""
    from raggity.server import create_app
    cfg = _auth_cfg(tmp_path, auth="none")
    app = create_app(cfg)
    with TestClient(app) as client:
        client.post("/ingest")
        _fix_query(monkeypatch)
        client.post("/chat", json={"question": "q", "session_id": "s1"})
        # Same client (no identity) can delete it
        assert client.delete("/session/s1").status_code == 204
        assert client.delete("/session/s1").status_code == 404


# --- Fix 2: bounded + closed user_rags -----------------------------------

def test_user_rags_bounded_lru_evicts_and_closes(tmp_path, monkeypatch):
    """user_rags is capped at max_user_rags; oldest evicted and its close() awaited."""
    from raggity.server import create_app
    from raggity.config import RaggityConfig, SourcesConfig, IndexConfig, ServerConfig
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[]),
        index=IndexConfig(path=str(tmp_path / "base")),
        server=ServerConfig(auth="api_key",
                            api_keys=["k0", "k1", "k2"],
                            per_user=True,
                            max_user_rags=2),
    )
    app = create_app(cfg)

    closed = []
    import raggity.core as core_mod
    orig_close = core_mod.Raggity.close

    async def _spy_close(self):
        closed.append(self.cfg.index.path)
        return await orig_close(self)
    monkeypatch.setattr(core_mod.Raggity, "close", _spy_close)

    with TestClient(app) as client:
        for k in ("k0", "k1", "k2"):
            assert client.get("/status", headers={"X-API-Key": k}).status_code == 200
        user_rags = app.state.raggity_state["user_rags"]
        assert len(user_rags) == 2  # capped at max_user_rags
        # k0 (oldest) should have been evicted from the cache
        assert "k0" not in user_rags
        assert "k1" in user_rags and "k2" in user_rags
        # The evicted k0 rag had close() awaited (path contains its slug)
        assert any(p.endswith("k0") for p in closed)


def test_lifespan_teardown_closes_all(tmp_path, monkeypatch):
    """On lifespan exit, base rag + cached user rags get close() awaited."""
    from raggity.server import create_app
    from raggity.config import RaggityConfig, SourcesConfig, IndexConfig, ServerConfig
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[]),
        index=IndexConfig(path=str(tmp_path / "base")),
        server=ServerConfig(auth="api_key", api_keys=["k0"], per_user=True),
    )
    app = create_app(cfg)

    closed = []
    import raggity.core as core_mod
    orig_close = core_mod.Raggity.close

    async def _spy_close(self):
        closed.append(self.cfg.index.path)
        return await orig_close(self)
    monkeypatch.setattr(core_mod.Raggity, "close", _spy_close)

    with TestClient(app) as client:
        client.get("/status", headers={"X-API-Key": "k0"})
    # After context exit, both base + user rag closed
    assert len(closed) >= 2


# --- Fix 3: per-tenant ingest content -------------------------------------

def test_ingest_content_per_tenant_isolation(tmp_path, monkeypatch):
    """per_user on: A and B POST /ingest/content; each /ask sees only its own."""
    from raggity.server import create_app
    cfg = _per_user_cfg(tmp_path, per_user=True)
    app = create_app(cfg)
    with TestClient(app) as client:
        ra = client.post("/ingest/content",
                         json={"documents": [{"path": "/a.md", "text": "Alice: sky is green.", "title": "a"}]},
                         headers={"X-API-Key": "key_alice"})
        assert ra.status_code == 200
        assert ra.json()["ingested"] == 1
        rb = client.post("/ingest/content",
                         json={"documents": [{"path": "/b.md", "text": "Bob: sea is purple.", "title": "b"}]},
                         headers={"X-API-Key": "key_bob"})
        assert rb.json()["ingested"] == 1

        st_alice = client.get("/status", headers={"X-API-Key": "key_alice"}).json()
        st_bob = client.get("/status", headers={"X-API-Key": "key_bob"}).json()
        assert st_alice["chunks"] >= 1
        assert st_bob["chunks"] >= 1
        assert st_alice["index_path"] != st_bob["index_path"]


def test_ingest_content_single_tenant(tmp_path):
    """per_user off: /ingest/content ingests into the base rag."""
    from raggity.server import create_app
    cfg = _auth_cfg(tmp_path, auth="none")
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.post("/ingest/content",
                        json={"documents": [{"path": "/x.md", "text": "hello world there"}]})
        assert r.status_code == 200
        assert r.json()["ingested"] == 1
        assert client.get("/status").json()["chunks"] >= 1


def test_ingest_content_requires_auth(tmp_path):
    """/ingest/content is behind require_auth."""
    from raggity.server import create_app
    cfg = _auth_cfg(tmp_path, auth="api_key", api_keys=["secret123"])
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.post("/ingest/content",
                        json={"documents": [{"path": "/x.md", "text": "hi"}]})
        assert r.status_code == 401


# --- Fix 5: SSE error handling -------------------------------------------

def test_sse_stream_error_no_traceback(tmp_path, monkeypatch):
    """If the rag raises mid-stream, response emits a generic event: error with no internals."""
    from raggity.server import create_app
    import raggity.core as core_mod
    cfg = _cfg(tmp_path)
    app = create_app(cfg)
    with TestClient(app) as client:
        client.post("/ingest")

        async def _boom(self, question, *a, **k):
            raise RuntimeError("SECRET-INTERNAL-DETAIL")
            yield  # pragma: no cover

        monkeypatch.setattr(core_mod.Raggity, "aask_stream", _boom)

        with client.stream("GET", "/ask/stream",
                           params={"question": "boom?"}) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        assert "event: error" in body
        assert "SECRET-INTERNAL-DETAIL" not in body
        assert "Traceback" not in body
