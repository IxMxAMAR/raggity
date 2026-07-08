"""Offline tests for `rag doctor`.  No real network / model loads."""
import raggity.doctor as doc
from raggity.config import GenerationConfig, RaggityConfig
from rich.console import Console


def _run(monkeypatch, **overrides):
    """Run doctor with all heavy checks stubbed to ok unless overridden.

    Returns (exit_code, rendered_text).
    """
    defaults = {
        "check_version": lambda: (doc.OK, "raggity x", ""),
        "check_config": lambda p: (doc.OK, "cfg", ""),
        "check_sources": lambda c: (doc.OK, "sources", ""),
        "check_index": lambda c: (doc.OK, "index", ""),
        "check_embedding": lambda c: (doc.OK, "emb", ""),
        "check_generation": lambda c: (doc.OK, "gen", ""),
        "check_index_writable": lambda c: (doc.OK, "writable", ""),
        "check_extras": lambda: [("extra:web", doc.INFO, "not installed", "")],
        "check_providers": lambda: [("provider:ollama", doc.INFO, "not found", "")],
    }
    defaults.update(overrides)
    for name, fn in defaults.items():
        monkeypatch.setattr(doc, name, fn)
    console = Console(record=True, width=200)
    code = doc.run_doctor(None, console)
    return code, console.export_text()


def test_doctor_all_ok_exit_0(monkeypatch):
    code, out = _run(monkeypatch)
    assert code == 0
    assert "[ok]" in out
    assert "[FAIL]" not in out


def test_doctor_fail_sets_exit_1_and_hint(monkeypatch):
    code, out = _run(monkeypatch,
                     check_embedding=lambda c: (doc.FAIL, "model boom", "install it"))
    assert code == 1
    assert "[FAIL]" in out
    assert "install it" in out


def test_doctor_warn_does_not_fail(monkeypatch):
    code, out = _run(monkeypatch,
                     check_sources=lambda c: (doc.WARN, "0 files", "add globs"))
    assert code == 0
    assert "[warn]" in out
    assert "add globs" in out


# ---------------------------------------------------------------------------
# check_generation: claude
# ---------------------------------------------------------------------------

def test_check_generation_claude_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
    status, detail, _ = doc.check_generation(RaggityConfig())
    assert status == doc.OK
    assert "claude" in detail


def test_check_generation_claude_with_cli(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(doc.shutil, "which", lambda n: r"C:\claude.exe" if n == "claude" else None)
    status, _, _ = doc.check_generation(RaggityConfig())
    assert status == doc.OK


def test_check_generation_claude_no_auth_warns(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(doc.shutil, "which", lambda n: None)
    status, _, hint = doc.check_generation(RaggityConfig())
    assert status == doc.WARN
    assert "claude login" in hint or "ANTHROPIC_API_KEY" in hint


# ---------------------------------------------------------------------------
# check_generation: ollama (mocked ensure/models)
# ---------------------------------------------------------------------------

def _ollama_cfg(model="gemma3"):
    return RaggityConfig(generation=GenerationConfig(backend="ollama", model=model))


def test_check_generation_ollama_reachable_model_present(monkeypatch):
    from raggity import providers
    monkeypatch.setattr(providers, "ensure_running", lambda *a, **k: True)
    monkeypatch.setattr(providers, "_models", lambda *a, **k: ["gemma3", "llama3"])
    status, detail, _ = doc.check_generation(_ollama_cfg("gemma3"))
    assert status == doc.OK
    assert "gemma3" in detail


def test_check_generation_ollama_reachable_model_missing(monkeypatch):
    from raggity import providers
    monkeypatch.setattr(providers, "ensure_running", lambda *a, **k: True)
    monkeypatch.setattr(providers, "_models", lambda *a, **k: ["llama3"])
    status, _, hint = doc.check_generation(_ollama_cfg("gemma3"))
    assert status == doc.WARN
    assert "ollama pull gemma3" in hint


def test_check_generation_ollama_unreachable_fail(monkeypatch):
    from raggity import providers
    monkeypatch.setattr(providers, "ensure_running", lambda *a, **k: False)
    status, _, hint = doc.check_generation(_ollama_cfg("gemma3"))
    assert status == doc.FAIL
    assert "Ollama running" in hint


# ---------------------------------------------------------------------------
# check_generation: external (readiness probe, NEVER auto-starts)
# ---------------------------------------------------------------------------

def _external_cfg(base_url="http://127.0.0.1:9999"):
    return RaggityConfig(generation=GenerationConfig(
        backend="external", model="rigma-model", base_url=base_url))


def test_check_generation_external_reachable_ok(monkeypatch):
    from raggity import providers
    monkeypatch.setattr(providers, "external_ready", lambda *a, **k: True)
    status, detail, _ = doc.check_generation(_external_cfg())
    assert status == doc.OK
    assert "external server reachable" in detail
    assert "http://127.0.0.1:9999" in detail


def test_check_generation_external_unreachable_fail(monkeypatch):
    from raggity import providers
    monkeypatch.setattr(providers, "external_ready", lambda *a, **k: False)
    status, detail, _ = doc.check_generation(_external_cfg())
    assert status == doc.FAIL
    assert "external server unreachable" in detail
    assert "never auto-starts" in detail
    assert "http://127.0.0.1:9999" in detail


def test_check_generation_external_missing_base_url_fail(monkeypatch):
    status, detail, hint = doc.check_generation(_external_cfg(base_url=None))
    assert status == doc.FAIL
    assert "base_url" in detail or "base_url" in hint


def test_check_generation_external_never_calls_ensure_running(monkeypatch):
    """doctor's external check must NOT touch the auto-start path."""
    from raggity import providers
    calls = []
    monkeypatch.setattr(providers, "ensure_running", lambda *a, **k: calls.append(a) or True)
    monkeypatch.setattr(providers, "external_ready", lambda *a, **k: True)
    doc.check_generation(_external_cfg())
    assert calls == []
