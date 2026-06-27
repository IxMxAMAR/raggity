from __future__ import annotations

import os
from abc import ABC, abstractmethod

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage

from .models import Answer, Chunk
from .prompts import (SYSTEM_PROMPT, ABSTAIN_MESSAGE, build_user_prompt,
                      verify_citations)
from .registry import register


class Answerer(ABC):
    @abstractmethod
    async def answer(self, question: str, chunks: list[Chunk]) -> Answer: ...


class ClaudeAgentAnswerer(Answerer):
    def __init__(self, model: str = "claude-opus-4-8", auth: str = "auto") -> None:
        self.model = model
        self.auth = auth

    def _options(self) -> ClaudeAgentOptions:
        if self.auth == "subscription":
            # Subscription-primary: pass os.environ minus ANTHROPIC_API_KEY so
            # the Agent SDK cannot fall back to a per-token API key and must use
            # the `claude login` subscription session.
            env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
            return ClaudeAgentOptions(
                system_prompt=SYSTEM_PROMPT,
                model=self.model,
                allowed_tools=[],
                permission_mode="dontAsk",
                env=env,
            )
        if self.auth == "api_key":
            # api_key mode: the SDK reads ANTHROPIC_API_KEY from the environment.
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "auth='api_key' but ANTHROPIC_API_KEY is not set. "
                    "Set the key, or use auth='subscription' after `claude login`."
                )
            return ClaudeAgentOptions(
                system_prompt=SYSTEM_PROMPT,
                model=self.model,
                allowed_tools=[],
                permission_mode="dontAsk",
            )
        # auth == "auto": leave env untouched — SDK resolves key-first, then
        # subscription session if no key is present.
        return ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            model=self.model,
            allowed_tools=[],
            permission_mode="dontAsk",
        )

    async def answer_stream(self, question: str, chunks: list[Chunk]):
        """Yield text-delta str items as they arrive, then a final Answer."""
        if not chunks:
            yield ABSTAIN_MESSAGE
            yield Answer(text=ABSTAIN_MESSAGE, citations=[], abstained=True)
            return
        prompt = build_user_prompt(question, chunks)
        opts = self._options()
        try:
            opts.include_partial_messages = True
        except Exception:
            pass
        parts: list[str] = []
        async for message in query(prompt=prompt, options=opts):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    text = getattr(block, "text", None)
                    if text:
                        parts.append(text)
                        yield text
        text = "".join(parts).strip()
        abstained = text == ABSTAIN_MESSAGE
        citations = [] if abstained else verify_citations(text, chunks)
        yield Answer(text=text, citations=citations, abstained=abstained)

    async def answer(self, question: str, chunks: list[Chunk]) -> Answer:
        if not chunks:
            return Answer(text=ABSTAIN_MESSAGE, citations=[], abstained=True)
        prompt = build_user_prompt(question, chunks)
        parts: list[str] = []
        async for message in query(prompt=prompt, options=self._options()):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    text = getattr(block, "text", None)
                    if text:
                        parts.append(text)
        text = "".join(parts).strip()
        abstained = text.strip() == ABSTAIN_MESSAGE
        citations = [] if abstained else verify_citations(text, chunks)
        return Answer(text=text, citations=citations, abstained=abstained)


register("answerer", "claude", "raggity.answerer:ClaudeAgentAnswerer")
