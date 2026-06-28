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
    import pytest
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        await llm.ClaudeProvider(auth="api_key").complete("s", "p")
