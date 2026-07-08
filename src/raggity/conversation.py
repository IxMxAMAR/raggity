from __future__ import annotations

# System prompt for the rolling-summary provider call: compress the oldest
# turns (optionally merging with any existing summary) into one short,
# third-person paragraph capturing facts/decisions/open threads.
_SUMMARY_SYSTEM_PROMPT = (
    "Compress the conversation excerpt below into a single third-person "
    "summary of at most 150 words, capturing the key facts, decisions, and "
    "open questions. If an EXISTING SUMMARY is provided, merge its content "
    "into the new summary rather than repeating it verbatim — produce ONE "
    "cohesive summary and output nothing else (no preamble, no labels)."
)


class Conversation:
    """Holds the turn history for a multi-turn chat session.

    Each turn is a ``(role, text)`` tuple where role ∈ ``{"user", "assistant"}``.
    """

    def __init__(self) -> None:
        self.turns: list[tuple[str, str]] = []
        # Rolling summary of turns compressed out of the window by
        # maybe_summarize(). Empty string = no summary yet.
        self.summary: str = ""

    def add(self, role: str, text: str) -> None:
        """Append a turn."""
        self.turns.append((role, text))

    async def maybe_summarize(self, provider, max_turns: int) -> None:
        """Compress the oldest turns into ``self.summary`` once the window overflows.

        No-op when ``max_turns <= 0`` (summarization disabled) or when the turn
        count has not yet exceeded ``max_turns``. Otherwise the oldest turns
        beyond the most-recent ``max_turns // 2`` are rendered to text and
        folded into ``self.summary`` via ONE ``provider.complete`` call, then
        dropped from ``self.turns``. If the provider call raises, the same
        oldest turns are still dropped (plain truncation) but ``self.summary``
        is left unchanged — summarization failure never raises into the
        calling chat turn.
        """
        if max_turns <= 0:
            return
        if len(self.turns) <= max_turns:
            return
        keep = max_turns // 2
        split = len(self.turns) - keep
        old_turns = self.turns[:split]
        kept_turns = self.turns[split:]
        old_text = "\n".join(f"{role}: {text}" for role, text in old_turns)
        if self.summary:
            prompt = (
                f"EXISTING SUMMARY:\n{self.summary}\n\n"
                f"NEW TURNS TO FOLD IN:\n{old_text}"
            )
        else:
            prompt = f"CONVERSATION EXCERPT:\n{old_text}"
        try:
            new_summary = await provider.complete(_SUMMARY_SYSTEM_PROMPT, prompt)
        except Exception:
            self.turns = kept_turns
            return
        self.summary = new_summary.strip()
        self.turns = kept_turns

    def recent(self, n: int) -> list[tuple[str, str]]:
        """Return the last *n* turns (or fewer if fewer exist)."""
        return self.turns[-n:] if n > 0 else []

    def retrieval_query(self, question: str) -> str:
        """Return a retrieval-oriented query string.

        When history exists, prepend the text of the last user turn to *question*
        so that the retrieval step has topical context from the ongoing conversation.
        If there is no prior user turn, the question is returned unchanged.
        """
        last_user_text: str | None = None
        for role, text in reversed(self.turns):
            if role == "user":
                last_user_text = text
                break
        if last_user_text is None:
            return question
        return f"{last_user_text} {question}"
