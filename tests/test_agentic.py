"""Agentic multi-round retrieval (`rag ask --agentic`).

The model orchestrates searches through an in-process SDK MCP tool
(`search_knowledge_base`). Tests mock the llm chokepoint (`raggity.llm.query`)
and drive the tool callback directly — no real Agent SDK / network calls. The
real `create_sdk_mcp_server` / `tool` are exercised (they only build an in-process
mcp Server object), which is how the test reaches the registered tool handler via
`options.mcp_servers[...]["instance"]._raggity_tools[0].handler`.
"""
import asyncio

import pytest

import raggity.agentic as agentic_mod
import raggity.llm as llm_mod
from raggity.config import (RaggityConfig, SourcesConfig, IndexConfig,
                            GenerationConfig)
from raggity.core import Raggity
from raggity.models import Answer, Chunk
from raggity.prompts import ABSTAIN_MESSAGE


class _Block:
    def __init__(self, text): self.text = text


class _AM:
    def __init__(self, text): self.content = [_Block(text)]


class _FakeRetriever:
    """Returns a distinct chunk-list per call (rounds), recording call args."""
    def __init__(self, rounds):
        self.rounds = rounds
        self.calls = []

    def retrieve(self, query, *, top_k=None, apply_sufficiency=True):
        self.calls.append({"query": query, "top_k": top_k,
                           "apply_sufficiency": apply_sufficiency})
        idx = min(len(self.calls) - 1, len(self.rounds) - 1)
        return list(self.rounds[idx])


def _hexid(c):
    return c * 16


def _chunk(hex_char, text, score=0.6):
    cid = _hexid(hex_char)
    return Chunk(text, f"{hex_char}.md", hex_char.upper(), hex_char.upper(),
                 0, cid, score=score)


def _rag(tmp_path, retriever, backend="claude", auth="auto"):
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[]),
        index=IndexConfig(path=str(tmp_path / "idx")),
        generation=GenerationConfig(backend=backend, auth=auth),
    )
    rag = Raggity(cfg)
    rag._retriever = retriever  # inject; bypasses embedder/store build
    return rag


# --- 1. CORE PROPERTY: multi-round accumulation + cross-round citation -------

async def test_multi_round_citation_accumulates_across_rounds(tmp_path, monkeypatch):
    c_a = _chunk("a", "alpha content about unrelated widgets", 0.6)
    c_b = _chunk("b", "backups run nightly to the NAS array", 0.7)
    retr = _FakeRetriever([[c_a], [c_b]])  # round1 -> c_a, round2 -> c_b
    rag = _rag(tmp_path, retr)

    async def _fake_query(prompt, options):
        cfg = options.mcp_servers["raggity_kb"]
        handler = cfg["instance"]._raggity_tools[0].handler
        # simulate two agentic rounds with different phrasings
        await handler({"query": "how are backups performed", "k": 6})
        await handler({"query": "NAS nightly schedule", "k": 6})
        # final answer cites a chunk retrieved ONLY in round 2
        yield _AM(f"Backups run nightly to the NAS [doc_1#{_hexid('b')}].")

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AM)

    ans = await rag.aask_agentic("how are backups done?")

    assert ans.abstained is False
    assert "NAS" in ans.text
    # the round-2 chunk must be resolvable -> proves accumulation across rounds
    cited = {c.chunk_id: c for c in ans.citations}
    assert _hexid("b") in cited
    assert cited[_hexid("b")].supported is True
    # tool was driven twice, raw retrieval (no sufficiency floor), k propagated
    assert len(retr.calls) == 2
    assert all(call["apply_sufficiency"] is False for call in retr.calls)
    assert all(call["top_k"] == 6 for call in retr.calls)


# --- 2. non-claude backend -> RuntimeError suggesting corrective ------------

async def test_non_claude_backend_raises(tmp_path, monkeypatch):
    rag = _rag(tmp_path, _FakeRetriever([[]]), backend="openai")
    with pytest.raises(RuntimeError, match="claude"):
        await rag.aask_agentic("anything")


# --- 3. abstention -----------------------------------------------------------

async def test_abstain_text_marks_answer_abstained(tmp_path, monkeypatch):
    rag = _rag(tmp_path, _FakeRetriever([[_chunk("a", "irrelevant")]]))

    async def _fake_query(prompt, options):
        yield _AM(ABSTAIN_MESSAGE)

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AM)

    ans = await rag.aask_agentic("who won in 1850?")
    assert ans.abstained is True
    assert ans.text == ABSTAIN_MESSAGE
    assert ans.citations == []


# --- 4. options hygiene: setting_sources, max_turns, allowed_tools ----------

async def test_options_hygiene(tmp_path, monkeypatch):
    rag = _rag(tmp_path, _FakeRetriever([[]]))
    captured = {}

    async def _fake_query(prompt, options):
        captured["opts"] = options
        yield _AM(ABSTAIN_MESSAGE)

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AM)

    await rag.aask_agentic("q")
    opts = captured["opts"]
    assert opts.setting_sources == []
    assert opts.max_turns == 8
    assert opts.allowed_tools == ["mcp__raggity_kb__search_knowledge_base"]
    assert "raggity_kb" in opts.mcp_servers


# --- 5. subscription env-strip reused from llm.base_options_kwargs ----------

async def test_subscription_strips_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "bearer-token")
    rag = _rag(tmp_path, _FakeRetriever([[]]), auth="subscription")
    captured = {}

    async def _fake_query(prompt, options):
        captured["opts"] = options
        yield _AM(ABSTAIN_MESSAGE)

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AM)

    await rag.aask_agentic("q")
    env = captured["opts"].env
    assert env is not None
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env


# --- 6. packed context carries citation tags the answerer format expects ----

async def test_tool_result_packs_citation_tags(tmp_path, monkeypatch):
    c_b = _chunk("b", "backups run nightly to the NAS array", 0.7)
    rag = _rag(tmp_path, _FakeRetriever([[c_b]]))
    captured = {}

    async def _fake_query(prompt, options):
        cfg = options.mcp_servers["raggity_kb"]
        handler = cfg["instance"]._raggity_tools[0].handler
        res = await handler({"query": "backups"})
        captured["res"] = res
        yield _AM(ABSTAIN_MESSAGE)

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AM)

    await rag.aask_agentic("q")
    text = captured["res"]["content"][0]["text"]
    assert f"[doc_1#{_hexid('b')}]" in text
    assert "[source: b.md]" in text
    assert "backups run nightly" in text
