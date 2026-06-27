import pytest
import os
from raggity.models import Chunk
from raggity.answerer import ClaudeAgentAnswerer
import raggity.answerer as answerer_mod


def _chunk(cid, text):
    return Chunk(text=text, source_path="notes/security.md", title="Security",
                 heading_path="Security", ordinal=0, chunk_id=cid)


class _Block:
    def __init__(self, text):
        self.text = text


class _AssistantMessage:
    def __init__(self, text):
        self.content = [_Block(text)]


async def _fake_query(prompt, options):
    # echo a grounded answer that cites the provided chunk
    yield _AssistantMessage("You rotated the API key on 2026-06-01 [doc_1#abcd1234].")


async def test_answer_streams_and_verifies(monkeypatch):
    monkeypatch.setattr(answerer_mod, "query", _fake_query)
    monkeypatch.setattr(answerer_mod, "AssistantMessage", _AssistantMessage)
    ans = await ClaudeAgentAnswerer(model="claude-opus-4-8").answer(
        "when did I rotate the key?",
        [_chunk("abcd1234ef", "rotated the API key on 2026-06-01")],
    )
    assert "2026-06-01" in ans.text
    assert ans.abstained is False
    assert any(c.supported for c in ans.citations)


async def test_answer_abstains_on_empty_chunks(monkeypatch):
    called = False

    async def _should_not_run(prompt, options):
        nonlocal called
        called = True
        yield _AssistantMessage("should not happen")

    monkeypatch.setattr(answerer_mod, "query", _should_not_run)
    ans = await ClaudeAgentAnswerer().answer("q", [])
    assert ans.abstained is True
    assert called is False


async def test_api_key_auth_raises_without_key(monkeypatch):
    """auth='api_key' with no ANTHROPIC_API_KEY in env must raise RuntimeError."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    async def _should_not_run(prompt, options):
        # _options() raises before query() is ever called
        raise AssertionError("query should not be reached")
        yield  # make it an async generator

    monkeypatch.setattr(answerer_mod, "query", _should_not_run)
    answerer = ClaudeAgentAnswerer(auth="api_key")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        await answerer.answer("q", [_chunk("x1", "some relevant text")])


async def test_answer_stream_yields_deltas_then_answer(monkeypatch):
    from raggity.models import Answer
    monkeypatch.setattr(answerer_mod, "query", _fake_query)
    monkeypatch.setattr(answerer_mod, "AssistantMessage", _AssistantMessage)
    items = []
    async for piece in ClaudeAgentAnswerer().answer_stream(
            "when did I rotate the key?",
            [_chunk("abcd1234ef", "rotated the API key on 2026-06-01")]):
        items.append(piece)
    assert isinstance(items[-1], Answer)
    assert any(isinstance(p, str) for p in items[:-1])
    assert "2026-06-01" in items[-1].text


async def test_answer_stream_empty_chunks_abstains(monkeypatch):
    from raggity.models import Answer
    called = False

    async def _no(prompt, options):
        nonlocal called
        called = True
        yield _AssistantMessage("x")

    monkeypatch.setattr(answerer_mod, "query", _no)
    out = [p async for p in ClaudeAgentAnswerer().answer_stream("q", [])]
    assert called is False
    assert isinstance(out[-1], Answer) and out[-1].abstained is True


async def test_subscription_auth_strips_api_key(monkeypatch):
    """auth='subscription' must pass env without ANTHROPIC_API_KEY to ClaudeAgentOptions,
    but must still include other env vars (not an empty dict)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    monkeypatch.setenv("HOME", "/home/testuser")  # sentinel — must survive in env
    captured_options = {}

    async def _capture_query(prompt, options):
        captured_options["opts"] = options
        yield _AssistantMessage("Answer without key.")

    monkeypatch.setattr(answerer_mod, "query", _capture_query)
    monkeypatch.setattr(answerer_mod, "AssistantMessage", _AssistantMessage)
    answerer = ClaudeAgentAnswerer(auth="subscription")
    await answerer.answer("q", [_chunk("y1", "some relevant text")])
    opts = captured_options["opts"]
    assert "ANTHROPIC_API_KEY" not in opts.env, "subscription must not expose API key"
    # env must be a real copy of os.environ (not an empty dict) so the SDK
    # process inherits PATH, HOME, etc.
    assert len(opts.env) > 1, "env should carry through other environment variables"
