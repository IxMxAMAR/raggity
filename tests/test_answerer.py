import pytest
import raggity.llm as llm
from raggity.answerer import ProviderAnswerer
from raggity.llm import ClaudeProvider


class _Block:
    def __init__(self, t): self.text = t


class _AM:
    def __init__(self, t): self.content = [_Block(t)]


def _chunk(cid, text):
    from raggity.models import Chunk
    return Chunk(text=text, source_path="notes/security.md", title="Security",
                 heading_path="Security", ordinal=0, chunk_id=cid)


async def test_answer_verifies_citations(monkeypatch):
    async def _fq(prompt, options):
        yield _AM("You rotated the API key on 2026-06-01 [doc_1#abcd1234].")
    monkeypatch.setattr(llm, "query", _fq)
    monkeypatch.setattr(llm, "AssistantMessage", _AM)
    ans = await ProviderAnswerer(ClaudeProvider()).answer(
        "when?", [_chunk("abcd1234ef", "rotated the API key on 2026-06-01")])
    assert "2026-06-01" in ans.text and any(c.supported for c in ans.citations)


async def test_answer_empty_abstains_no_call(monkeypatch):
    called = False

    async def _fq(prompt, options):
        nonlocal called
        called = True
        yield _AM("x")

    monkeypatch.setattr(llm, "query", _fq)
    ans = await ProviderAnswerer(ClaudeProvider()).answer("q", [])
    assert ans.abstained and called is False


async def test_answer_stream_yields_then_answer(monkeypatch):
    from raggity.models import Answer

    async def _fq(prompt, options):
        yield _AM("Rotated on 2026-06-01 [doc_1#abcd1234].")

    monkeypatch.setattr(llm, "query", _fq)
    monkeypatch.setattr(llm, "AssistantMessage", _AM)
    items = [p async for p in ProviderAnswerer(ClaudeProvider()).answer_stream(
        "q", [_chunk("abcd1234ef", "rotated the API key on 2026-06-01")])]
    assert isinstance(items[-1], Answer) and any(isinstance(x, str) for x in items[:-1])
