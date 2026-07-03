"""Offline tests for local-provider discovery + auto-start.

All network / process / PATH access is monkeypatched: no real urllib, subprocess,
or shutil.which calls reach the system.
"""
import json
import urllib.error

import raggity.providers as prov


class _FakeResp:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _urlopen_ok(mapping):
    """Return a fake urlopen that serves *mapping* {url_substr: payload}, else refuses."""
    def _fake(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        for frag, payload in mapping.items():
            if frag in url:
                return _FakeResp(payload)
        raise urllib.error.URLError("connection refused")
    return _fake


# ---------------------------------------------------------------------------
# discover()
# ---------------------------------------------------------------------------

def test_discover_running_with_models(monkeypatch):
    """Ollama reachable (version ok) + tags returns installed models."""
    mapping = {
        "/api/version": {"version": "0.1.0"},
        "/api/tags": {"models": [{"name": "gemma3"}, {"name": "llama3"}]},
    }
    monkeypatch.setattr(prov.urllib.request, "urlopen", _urlopen_ok(mapping))
    monkeypatch.setattr(prov.shutil, "which", lambda n: None)
    st = {s.name: s for s in prov.discover()}["ollama"]
    assert st.running is True
    assert "gemma3" in st.models and "llama3" in st.models


def test_discover_installed_not_running(monkeypatch):
    """Binary present but server refuses connection => installed, not running."""
    monkeypatch.setattr(prov.urllib.request, "urlopen", _urlopen_ok({}))  # everything refuses
    monkeypatch.setattr(prov.shutil, "which", lambda n: r"C:\ollama.exe" if n == "ollama" else None)
    st = {s.name: s for s in prov.discover()}["ollama"]
    assert st.running is False
    assert st.installed is True
    assert st.auto_startable is True  # exe found -> can be started


def test_discover_not_found(monkeypatch):
    monkeypatch.setattr(prov.urllib.request, "urlopen", _urlopen_ok({}))
    monkeypatch.setattr(prov.shutil, "which", lambda n: None)
    monkeypatch.setattr(prov.os.path, "isfile", lambda p: False)
    monkeypatch.setattr(prov.os.path, "exists", lambda p: False)
    st = {s.name: s for s in prov.discover()}["ollama"]
    assert st.running is False and st.installed is False and st.auto_startable is False


# ---------------------------------------------------------------------------
# ensure_running()
# ---------------------------------------------------------------------------

def test_ensure_running_already_up_no_spawn(monkeypatch):
    monkeypatch.setattr(prov, "_reachable", lambda name, base: True)
    spawned = []
    monkeypatch.setattr(prov, "_spawn_detached", lambda cmd: spawned.append(cmd))
    assert prov.ensure_running("ollama") is True
    assert spawned == []


def test_ensure_running_down_with_binary_spawns_and_polls(monkeypatch):
    state = {"up": False}
    monkeypatch.setattr(prov, "_reachable", lambda name, base: state["up"])
    monkeypatch.setattr(prov, "_exe", lambda spec: r"C:\ollama.exe")

    spawned = []

    def _spawn(cmd):
        spawned.append(cmd)
        state["up"] = True  # server becomes reachable after spawn

    monkeypatch.setattr(prov, "_spawn_detached", _spawn)
    monkeypatch.setattr(prov.time, "sleep", lambda s: None)
    assert prov.ensure_running("ollama", timeout=5) is True
    assert spawned and spawned[0][0] == r"C:\ollama.exe" and spawned[0][1] == "serve"


def test_ensure_running_down_no_binary_returns_false(monkeypatch):
    monkeypatch.setattr(prov, "_reachable", lambda name, base: False)
    monkeypatch.setattr(prov, "_exe", lambda spec: None)
    spawned = []
    monkeypatch.setattr(prov, "_spawn_detached", lambda cmd: spawned.append(cmd))
    assert prov.ensure_running("ollama") is False
    assert spawned == []


def test_ensure_running_discovery_only_provider(monkeypatch):
    """llamacpp has no auto_start => ensure_running never spawns."""
    monkeypatch.setattr(prov, "_reachable", lambda name, base: False)
    spawned = []
    monkeypatch.setattr(prov, "_spawn_detached", lambda cmd: spawned.append(cmd))
    assert prov.ensure_running("llamacpp") is False
    assert spawned == []


def test_is_local():
    assert prov.is_local("http://localhost:11434/v1")
    assert prov.is_local("http://127.0.0.1:1234/v1")
    assert not prov.is_local("https://api.openai.com/v1")
    assert not prov.is_local(None)
