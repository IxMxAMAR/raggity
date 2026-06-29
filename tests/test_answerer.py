import pytest
import raggity.llm as llm
from raggity.answerer import ProviderAnswerer
from raggity.llm import ClaudeProvider
from raggity.prompts import _TAG_PREFIX_LEN

# Use a chunk_id long enough to supply the full citation prefix (16 hex chars).
_CID = "abcd1234ef567890aabb"
_PREFIX = _CID[:_TAG_PREFIX_LEN]   # "abcd1234ef567890"


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
        yield _AM(f"You rotated the API key on 2026-06-01 [doc_1#{_PREFIX}].")
    monkeypatch.setattr(llm, "query", _fq)
    monkeypatch.setattr(llm, "AssistantMessage", _AM)
    ans = await ProviderAnswerer(ClaudeProvider()).answer(
        "when?", [_chunk(_CID, "rotated the API key on 2026-06-01")])
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
        yield _AM(f"Rotated on 2026-06-01 [doc_1#{_PREFIX}].")

    monkeypatch.setattr(llm, "query", _fq)
    monkeypatch.setattr(llm, "AssistantMessage", _AM)
    items = [p async for p in ProviderAnswerer(ClaudeProvider()).answer_stream(
        "q", [_chunk(_CID, "rotated the API key on 2026-06-01")])]
    assert isinstance(items[-1], Answer) and any(isinstance(x, str) for x in items[:-1])


async def test_answer_with_history_includes_history_in_prompt(monkeypatch):
    """history param must be passed into build_user_prompt; prompt should contain history text."""
    captured = {}

    async def _fq(prompt, options):
        # prompt is the string passed to query()
        captured["user_prompt"] = prompt
        yield _AM(f"The answer [doc_1#{_PREFIX}].")

    monkeypatch.setattr(llm, "query", _fq)
    monkeypatch.setattr(llm, "AssistantMessage", _AM)
    history = [("user", "previous question"), ("assistant", "previous answer")]
    await ProviderAnswerer(ClaudeProvider()).answer(
        "follow up", [_chunk(_CID, "some relevant text")], history=history)
    assert "previous question" in captured["user_prompt"]
    assert "CONVERSATION SO FAR" in captured["user_prompt"]


async def test_answer_stream_with_history_includes_history_in_prompt(monkeypatch):
    """answer_stream with history must fold history into prompt."""
    captured = {}

    async def _fq(prompt, options):
        captured["user_prompt"] = prompt
        yield _AM(f"The answer [doc_1#{_PREFIX}].")

    monkeypatch.setattr(llm, "query", _fq)
    monkeypatch.setattr(llm, "AssistantMessage", _AM)
    history = [("user", "what is X"), ("assistant", "X is Y")]
    items = [p async for p in ProviderAnswerer(ClaudeProvider()).answer_stream(
        "tell me more", [_chunk(_CID, "X is a concept")], history=history)]
    assert "what is X" in captured["user_prompt"]


async def test_answer_history_none_identical_to_no_history(monkeypatch):
    """history=None must NOT alter the prompt vs no history arg."""
    prompts_seen = []

    async def _fq(prompt, options):
        prompts_seen.append(prompt)
        yield _AM(f"Some answer [doc_1#{_PREFIX}].")

    monkeypatch.setattr(llm, "query", _fq)
    monkeypatch.setattr(llm, "AssistantMessage", _AM)
    chunk = _chunk(_CID, "some text")
    answerer = ProviderAnswerer(ClaudeProvider())
    await answerer.answer("q", [chunk])
    await answerer.answer("q", [chunk], history=None)
    assert prompts_seen[0] == prompts_seen[1]
