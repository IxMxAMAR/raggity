from __future__ import annotations

import pytest
from raggity.conversation import Conversation


def test_conversation_starts_empty():
    c = Conversation()
    assert c.turns == []


def test_conversation_add_user_turn():
    c = Conversation()
    c.add("user", "hello")
    assert c.turns == [("user", "hello")]


def test_conversation_add_multiple_turns():
    c = Conversation()
    c.add("user", "q1")
    c.add("assistant", "a1")
    assert len(c.turns) == 2
    assert c.turns[0] == ("user", "q1")
    assert c.turns[1] == ("assistant", "a1")


def test_conversation_recent_returns_last_n():
    c = Conversation()
    c.add("user", "q1")
    c.add("assistant", "a1")
    c.add("user", "q2")
    c.add("assistant", "a2")
    recent = c.recent(2)
    assert len(recent) == 2
    assert recent[0] == ("user", "q2")
    assert recent[1] == ("assistant", "a2")


def test_conversation_recent_clamps_to_available():
    c = Conversation()
    c.add("user", "q1")
    recent = c.recent(10)
    assert recent == [("user", "q1")]


def test_conversation_recent_empty_when_no_turns():
    c = Conversation()
    assert c.recent(3) == []


def test_conversation_recent_and_retrieval_query():
    """From task brief."""
    c = Conversation()
    c.add("user", "what GPUs do I have?")
    c.add("assistant", "RTX 5090 and RX 9070 XT")
    assert c.recent(2)[-1][0] == "assistant"
    q = c.retrieval_query("and the dev box?")
    assert "dev box" in q and "GPUs" in q   # last user turn folded into retrieval query


def test_retrieval_query_no_history_returns_question_unchanged():
    c = Conversation()
    q = c.retrieval_query("what is the capital of France?")
    assert q == "what is the capital of France?"


def test_retrieval_query_with_history_prepends_last_user_turn():
    c = Conversation()
    c.add("user", "tell me about neural networks")
    c.add("assistant", "Neural networks are...")
    c.add("user", "how many layers?")
    c.add("assistant", "It depends on the architecture")
    q = c.retrieval_query("what about transformers?")
    assert "how many layers" in q
    assert "transformers" in q


def test_retrieval_query_only_assistant_turns_returns_question():
    """Edge case: if no user turns exist yet (shouldn't happen normally), just return question."""
    c = Conversation()
    c.add("assistant", "hello, how can I help?")
    q = c.retrieval_query("hi there")
    assert "hi there" in q


# --- prompts history tests ---

def test_build_user_prompt_with_history():
    """From task brief."""
    from raggity.prompts import build_user_prompt
    from raggity.models import Chunk
    ch = Chunk(text="x", source_path="a.md", title="A", heading_path="A", ordinal=0, chunk_id="c1000000")
    p = build_user_prompt("follow up?", [ch], history=[("user", "first q"), ("assistant", "first a")])
    assert "first q" in p and "follow up?" in p and "CONTEXT" in p


def test_build_user_prompt_no_history_unchanged():
    """history=None must produce byte-identical output to the original."""
    from raggity.prompts import build_user_prompt
    from raggity.models import Chunk
    ch = Chunk(text="x", source_path="a.md", title="A", heading_path="A", ordinal=0, chunk_id="c1000000")
    p_none = build_user_prompt("q?", [ch], history=None)
    p_default = build_user_prompt("q?", [ch])
    assert p_none == p_default


def test_build_user_prompt_with_history_has_conversation_block():
    from raggity.prompts import build_user_prompt
    from raggity.models import Chunk
    ch = Chunk(text="some context", source_path="b.md", title="B", heading_path="B",
               ordinal=0, chunk_id="d2000000")
    history = [("user", "first question"), ("assistant", "first answer")]
    p = build_user_prompt("second question", [ch], history=history)
    assert "CONVERSATION SO FAR" in p
    assert "first question" in p
    assert "first answer" in p
    assert "second question" in p
    assert "CONTEXT" in p


def test_build_user_prompt_history_comes_before_context():
    from raggity.prompts import build_user_prompt
    from raggity.models import Chunk
    ch = Chunk(text="some context", source_path="b.md", title="B", heading_path="B",
               ordinal=0, chunk_id="d2000000")
    history = [("user", "q1")]
    p = build_user_prompt("q2", [ch], history=history)
    conv_pos = p.index("CONVERSATION SO FAR")
    ctx_pos = p.index("CONTEXT")
    assert conv_pos < ctx_pos, "history block must precede CONTEXT block"


def test_build_user_prompt_renders_summary_role_as_leading_line():
    """A ('summary', text) history entry renders as a synthetic leading line,
    not the raw 'summary: ...' role rendering used for user/assistant."""
    from raggity.prompts import build_user_prompt
    from raggity.models import Chunk
    ch = Chunk(text="x", source_path="a.md", title="A", heading_path="A",
               ordinal=0, chunk_id="c1000000")
    history = [("summary", "Earlier they discussed GPUs."), ("user", "and the dev box?")]
    p = build_user_prompt("follow up?", [ch], history=history)
    assert "Earlier conversation summary: Earlier they discussed GPUs." in p
    assert "user: and the dev box?" in p
    lines = p.splitlines()
    assert "summary: Earlier they discussed GPUs." not in lines


# ---------------------------------------------------------------------------
# Task 6b: rolling conversation-summary memory (maybe_summarize)
# ---------------------------------------------------------------------------

import asyncio


class _FakeSummaryProvider:
    def __init__(self, response: str = "A summary."):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        return self.response


def _fill(c: Conversation, n: int) -> None:
    for i in range(n):
        c.add("user" if i % 2 == 0 else "assistant", f"t{i}")


def test_maybe_summarize_below_threshold_no_provider_call():
    c = Conversation()
    _fill(c, 4)
    prov = _FakeSummaryProvider()
    asyncio.run(c.maybe_summarize(prov, 4))  # 4 turns, max_turns=4: not yet over
    assert prov.calls == []
    assert len(c.turns) == 4
    assert c.summary == ""


def test_maybe_summarize_crossing_threshold_one_call_drops_old_turns():
    c = Conversation()
    _fill(c, 5)  # exceeds max_turns=4
    prov = _FakeSummaryProvider("Discussed backups and GPUs.")
    asyncio.run(c.maybe_summarize(prov, 4))
    assert len(prov.calls) == 1
    assert c.summary == "Discussed backups and GPUs."
    # keep = max_turns // 2 = 2 most-recent turns retained
    assert c.turns == [("assistant", "t3"), ("user", "t4")]


def test_maybe_summarize_second_crossing_merges_prompt_contains_old_summary():
    c = Conversation()
    c.summary = "Earlier: discussed GPUs."
    _fill(c, 5)
    prov = _FakeSummaryProvider("Merged: GPUs then backups.")
    asyncio.run(c.maybe_summarize(prov, 4))
    assert len(prov.calls) == 1
    _system, prompt = prov.calls[0]
    assert "Earlier: discussed GPUs." in prompt  # old summary folded into the provider call
    assert c.summary == "Merged: GPUs then backups."
    assert len(c.turns) == 2


def test_maybe_summarize_provider_failure_falls_back_to_truncation():
    c = Conversation()
    _fill(c, 5)

    class _RaisingProvider:
        async def complete(self, system, prompt):
            raise RuntimeError("provider unavailable")

    asyncio.run(c.maybe_summarize(_RaisingProvider(), 4))  # must not raise
    assert c.summary == ""  # left unchanged since the call never succeeded
    assert len(c.turns) == 2  # still truncated to the recent window


def test_maybe_summarize_zero_max_turns_never_summarizes():
    c = Conversation()
    _fill(c, 50)
    prov = _FakeSummaryProvider()
    asyncio.run(c.maybe_summarize(prov, 0))
    assert prov.calls == []
    assert len(c.turns) == 50
    assert c.summary == ""


def test_maybe_summarize_recent_window_intact_after_summarization():
    """The most-recent turns kept after summarization must equal recent(keep)."""
    c = Conversation()
    _fill(c, 9)
    prov = _FakeSummaryProvider("Summary.")
    asyncio.run(c.maybe_summarize(prov, 6))  # keep = 3
    assert c.turns == [("user", "t6"), ("assistant", "t7"), ("user", "t8")]
