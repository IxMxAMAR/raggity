from __future__ import annotations


class Conversation:
    """Holds the turn history for a multi-turn chat session.

    Each turn is a ``(role, text)`` tuple where role ∈ ``{"user", "assistant"}``.
    """

    def __init__(self) -> None:
        self.turns: list[tuple[str, str]] = []

    def add(self, role: str, text: str) -> None:
        """Append a turn."""
        self.turns.append((role, text))

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
