import os
import pytest
import raggity.llm as llm


class _Block:
    def __init__(self, t): self.text = t
class _AM:
    def __init__(self, t): self.content = [_Block(t)]


async def test_claude_provider_complete(monkeypatch):
    async def _fake_query(prompt, options):
        yield _AM("hello "); yield _AM("world")
    monkeypatch.setattr(llm, "query", _fake_query)
    monkeypatch.setattr(llm, "AssistantMessage", _AM)
    p = llm.ClaudeProvider(model="claude-opus-4-8", auth="auto")
    assert (await p.complete("sys", "hi")).strip() == "hello world"


async def test_claude_provider_stream(monkeypatch):
    async def _fake_query(prompt, options):
        yield _AM("a"); yield _AM("b")
    monkeypatch.setattr(llm, "query", _fake_query)
    monkeypatch.setattr(llm, "AssistantMessage", _AM)
    out = [t async for t in llm.ClaudeProvider().stream("sys", "hi")]
    assert out == ["a", "b"]


def test_build_provider_default_claude():
    from raggity.config import GenerationConfig
    assert isinstance(llm.build_provider(GenerationConfig()), llm.ClaudeProvider)


async def test_claude_api_key_mode_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    async def _fake_query(prompt, options):
        yield _AM("x")
    monkeypatch.setattr(llm, "query", _fake_query); monkeypatch.setattr(llm, "AssistantMessage", _AM)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        await llm.ClaudeProvider(auth="api_key").complete("s", "p")


# Fix 4: subscription mode strips ANTHROPIC_AUTH_TOKEN (and ANTHROPIC_API_KEY)
def test_subscription_strips_auth_token(monkeypatch):
    """subscription auth must exclude ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN from env."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "bearer-token")
    p = llm.ClaudeProvider(auth="subscription")
    opts = p._options("sys")
    env = opts.env
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env


# Fix 1: LLMProvider has optional aclose no-op; ClaudeProvider aclose is no-op
async def test_llm_provider_aclose_noop():
    """aclose() on ClaudeProvider must be callable and be a no-op (no error)."""
    p = llm.ClaudeProvider()
    await p.aclose()  # must not raise


# Fix 1: aclose is on LLMProvider base so any provider is callable uniformly
def test_llm_provider_base_has_aclose():
    """LLMProvider base class must expose aclose() so callers are uniform."""
    import inspect
    # aclose should exist on the base class
    assert hasattr(llm.LLMProvider, "aclose")


# Perf (v0.9.0 Task 1): importing raggity.llm must not eagerly pull in claude_agent_sdk
# (its __init__ drags in the full mcp package incl. server stack -> ~344ms). Run in a
# subprocess for a clean sys.modules slate, robust regardless of test execution order.
def test_import_llm_does_not_import_claude_agent_sdk():
    import subprocess
    import sys
    code = (
        "import sys\n"
        "import raggity.llm\n"
        "leaked = [m for m in sys.modules if m == 'claude_agent_sdk' or m.startswith('claude_agent_sdk.')]\n"
        "assert not leaked, f'claude_agent_sdk leaked into sys.modules: {leaked}'\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


# Perf (v0.9.0 Task 1): _options() should isolate the SDK call from user/project
# settings and CLAUDE.md (setting_sources=[]) and cap single-shot turns (max_turns=1).
# Uses the real ClaudeAgentOptions (SDK installed in the test venv) since no query
# patch is applied here, exercising the real _ensure_sdk() path.
def test_options_sets_setting_sources_empty_and_max_turns_one():
    p = llm.ClaudeProvider()
    opts = p._options("sys")
    assert opts.setting_sources == []
    assert opts.max_turns == 1


# Perf (v0.9.0 Task 5b): true token streaming. When the SDK forwards incremental
# StreamEvent text deltas, the provider yields them AND suppresses the final
# AssistantMessage snapshot (which repeats the same text) to avoid duplication.
class _SE:
    """Minimal StreamEvent-shaped stub (has an `.event` dict like the real one)."""
    def __init__(self, event): self.event = event


def _text_delta(t):
    return _SE({"type": "content_block_delta",
                "delta": {"type": "text_delta", "text": t}})


async def test_stream_yields_token_deltas_and_suppresses_snapshot(monkeypatch):
    async def _fake_query(prompt, options):
        yield _text_delta("Hel")
        yield _text_delta("lo")
        yield _AM("Hello")  # final complete snapshot — must NOT be re-yielded
    monkeypatch.setattr(llm, "query", _fake_query)
    monkeypatch.setattr(llm, "AssistantMessage", _AM)
    monkeypatch.setattr(llm, "StreamEvent", _SE)
    out = [t async for t in llm.ClaudeProvider().stream("sys", "hi")]
    assert out == ["Hel", "lo"]


async def test_stream_ignores_non_text_stream_events(monkeypatch):
    async def _fake_query(prompt, options):
        # A non-text delta (e.g. thinking) is ignored; text still streams; snapshot skipped.
        yield _SE({"type": "content_block_delta",
                   "delta": {"type": "thinking_delta", "thinking": "..."}})
        yield _text_delta("hi")
        yield _AM("hi")
    monkeypatch.setattr(llm, "query", _fake_query)
    monkeypatch.setattr(llm, "AssistantMessage", _AM)
    monkeypatch.setattr(llm, "StreamEvent", _SE)
    out = [t async for t in llm.ClaudeProvider().stream("sys", "hi")]
    assert out == ["hi"]


async def test_stream_am_only_fallback_unchanged(monkeypatch):
    # No StreamEvents in the stream → fall back to yielding AssistantMessage text
    # (keeps every existing AM-only mock working exactly as before).
    async def _fake_query(prompt, options):
        yield _AM("a"); yield _AM("b")
    monkeypatch.setattr(llm, "query", _fake_query)
    monkeypatch.setattr(llm, "AssistantMessage", _AM)
    out = [t async for t in llm.ClaudeProvider().stream("sys", "hi")]
    assert out == ["a", "b"]
