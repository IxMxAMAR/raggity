import pytest
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
