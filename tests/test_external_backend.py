"""backend=external: externally-managed OpenAI-compatible server (Rigma owns it).

raggity must NEVER start/stop it. Readiness is a lazy health->/v1/models probe,
cached per provider instance; both dead => clean RuntimeError naming the URL.
"""
import types

import pytest

import raggity.llm as llm
import raggity.llm_openai as lo
import raggity.providers as providers
from raggity.config import GenerationConfig


# --- helpers ---------------------------------------------------------------

class _Delta:
    def __init__(self, c): self.content = c
class _Choice:
    def __init__(self, c): self.delta = _Delta(c); self.message = types.SimpleNamespace(content=c)
class _Chunk:
    def __init__(self, c): self.choices = [_Choice(c)]


class _FakeCompletions:
    def __init__(self): self.calls = 0
    async def create(self, model, messages, stream=False, **kw):
        self.calls += 1
        if stream:
            async def gen():
                yield _Chunk("hi")
            return gen()
        return types.SimpleNamespace(choices=[_Choice("hi")])


class _FakeClient:
    def __init__(self, **kw):
        self.chat = types.SimpleNamespace(completions=_FakeCompletions())
    async def close(self): pass


def _ext_provider(monkeypatch, base_url="http://127.0.0.1:9999"):
    monkeypatch.setattr(lo, "AsyncOpenAI", _FakeClient)
    cfg = GenerationConfig(backend="external", model="rigma-model", base_url=base_url)
    return lo.OpenAICompatProvider.from_config(cfg)


# --- routing / construction ------------------------------------------------

def test_build_provider_routes_external(monkeypatch):
    monkeypatch.setattr(lo, "AsyncOpenAI", _FakeClient)
    cfg = GenerationConfig(backend="external", model="m", base_url="http://127.0.0.1:9999")
    p = llm.build_provider(cfg)
    assert isinstance(p, lo.OpenAICompatProvider)
    assert p._external is True


def test_external_requires_base_url(monkeypatch):
    """Missing base_url => clear construction error."""
    monkeypatch.setattr(lo, "AsyncOpenAI", _FakeClient)
    cfg = GenerationConfig(backend="external", model="m", base_url=None)
    with pytest.raises(RuntimeError, match="base_url"):
        lo.OpenAICompatProvider.from_config(cfg)


def test_external_never_resolves_ensure_provider(monkeypatch):
    """External must NOT carry an auto-start hook (spawn path structurally off)."""
    p = _ext_provider(monkeypatch)
    assert p._ensure_provider is None


def test_external_local_needs_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = _ext_provider(monkeypatch, base_url="http://127.0.0.1:9999")  # must not raise
    assert p is not None


def test_external_remote_no_key_is_keyless_not_required(monkeypatch):
    """Non-local external URL without key => keyless, does NOT hard-require a key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = _ext_provider(monkeypatch, base_url="https://rigma.example.com/v1")  # must not raise
    assert p is not None


# --- readiness probe: health / v1/models fallback --------------------------

def test_external_ready_health_ok_skips_models(monkeypatch):
    """/health 200 => ready, WITHOUT touching /v1/models."""
    seen = []
    def _fake_get_json(url, timeout=2.0):
        seen.append(url)
        if url.endswith("/health"):
            return {"status": "ok"}
        raise AssertionError(f"should not probe {url}")
    monkeypatch.setattr(providers, "_get_json", _fake_get_json)
    assert providers.external_ready("http://127.0.0.1:9999") is True
    assert seen == ["http://127.0.0.1:9999/health"]


def test_external_ready_falls_back_to_models(monkeypatch):
    """No /health, but /v1/models 200 => ready via fallback."""
    seen = []
    def _fake_get_json(url, timeout=2.0):
        seen.append(url)
        if url.endswith("/health"):
            raise OSError("no health endpoint")
        if url.endswith("/v1/models"):
            return {"data": []}
        raise AssertionError(url)
    monkeypatch.setattr(providers, "_get_json", _fake_get_json)
    assert providers.external_ready("http://127.0.0.1:9999") is True
    assert seen == ["http://127.0.0.1:9999/health", "http://127.0.0.1:9999/v1/models"]


def test_external_ready_both_dead_returns_false(monkeypatch):
    monkeypatch.setattr(providers, "_get_json",
                        lambda url, timeout=2.0: (_ for _ in ()).throw(OSError("dead")))
    assert providers.external_ready("http://127.0.0.1:9999") is False


def test_external_ready_handles_v1_suffixed_base_url(monkeypatch):
    """base_url ending in /v1 still probes root/health + root/v1/models."""
    seen = []
    def _fake_get_json(url, timeout=2.0):
        seen.append(url)
        raise OSError("dead")
    monkeypatch.setattr(providers, "_get_json", _fake_get_json)
    providers.external_ready("http://127.0.0.1:9999/v1")
    assert seen == ["http://127.0.0.1:9999/health", "http://127.0.0.1:9999/v1/models"]


# --- provider integration: dead URL, zero spawn, once-per-instance ---------

async def test_external_dead_url_raises_clean_error_no_spawn(monkeypatch):
    """rag ask-level call (complete) on a dead external server:
    clean RuntimeError naming the URL AND zero auto-start attempts."""
    spawn_calls = []
    monkeypatch.setattr(providers, "ensure_running",
                        lambda *a, **k: spawn_calls.append(a) or True)
    monkeypatch.setattr(providers, "_spawn_detached",
                        lambda *a, **k: spawn_calls.append(("spawn", a)))
    monkeypatch.setattr(providers, "external_ready", lambda *a, **k: False)

    p = _ext_provider(monkeypatch, base_url="http://127.0.0.1:9999")
    with pytest.raises(RuntimeError) as ei:
        await p.complete("s", "u")
    msg = str(ei.value)
    assert "http://127.0.0.1:9999" in msg
    assert "external" in msg and "never" in msg
    assert spawn_calls == []  # ZERO spawn / ensure attempts


async def test_external_readiness_runs_once_per_instance(monkeypatch):
    """Probe fires exactly once across two complete() calls (cached on success)."""
    probes = []
    monkeypatch.setattr(providers, "external_ready",
                        lambda *a, **k: probes.append(a) or True)
    p = _ext_provider(monkeypatch, base_url="http://127.0.0.1:9999")
    await p.complete("s", "u")
    await p.complete("s", "u")
    assert len(probes) == 1
    assert p._client.chat.completions.calls == 2  # both requests went through


async def test_external_stream_gated_by_readiness(monkeypatch):
    monkeypatch.setattr(providers, "external_ready", lambda *a, **k: False)
    p = _ext_provider(monkeypatch, base_url="http://127.0.0.1:9999")
    with pytest.raises(RuntimeError, match="9999"):
        async for _ in p.stream("s", "u"):
            pass


# --- ollama behavior UNCHANGED (regression guard) --------------------------

def test_ollama_still_resolves_ensure_provider(monkeypatch):
    """Existing auto-start wiring for ollama is untouched by the external branch."""
    monkeypatch.setattr(lo, "AsyncOpenAI", _FakeClient)
    cfg = GenerationConfig(backend="ollama", model="llama3.1")
    p = lo.OpenAICompatProvider.from_config(cfg)
    assert p._ensure_provider == "ollama"
    assert p._external is False
