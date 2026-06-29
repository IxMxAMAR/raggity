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
