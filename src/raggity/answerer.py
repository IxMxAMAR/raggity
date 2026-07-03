from __future__ import annotations

from abc import ABC, abstractmethod

from .llm import LLMProvider
from .models import Answer, Chunk
from .prompts import (SYSTEM_PROMPT, ABSTAIN_MESSAGE, build_user_prompt,
                      verify_citations)
from .registry import register


class Answerer(ABC):
    @abstractmethod
    async def answer(
        self,
        question: str,
        chunks: list[Chunk],
        history: list[tuple[str, str]] | None = None,
    ) -> Answer: ...

    @abstractmethod
    def answer_stream(
        self,
        question: str,
        chunks: list[Chunk],
        history: list[tuple[str, str]] | None = None,
    ): ...


class ProviderAnswerer(Answerer):
    def __init__(self, provider: LLMProvider, system_prompt: str = SYSTEM_PROMPT) -> None:
        self.provider = provider
        self.system_prompt = system_prompt

    async def answer(
        self,
        question: str,
        chunks: list[Chunk],
        history: list[tuple[str, str]] | None = None,
    ) -> Answer:
        if not chunks:
            return Answer(text=ABSTAIN_MESSAGE, citations=[], abstained=True)
        prompt = build_user_prompt(question, chunks, history=history)
        text = (await self.provider.complete(self.system_prompt, prompt)).strip()
        abst = text == ABSTAIN_MESSAGE
        return Answer(text=text, citations=[] if abst else verify_citations(text, chunks), abstained=abst)

    async def answer_stream(
        self,
        question: str,
        chunks: list[Chunk],
        history: list[tuple[str, str]] | None = None,
    ):
        """Yield text-delta str items as they arrive, then a final Answer."""
        if not chunks:
            yield ABSTAIN_MESSAGE
            yield Answer(text=ABSTAIN_MESSAGE, citations=[], abstained=True)
            return
        prompt = build_user_prompt(question, chunks, history=history)
        parts: list[str] = []
        async for t in self.provider.stream(self.system_prompt, prompt):
            parts.append(t)
            yield t
        text = "".join(parts).strip()
        abst = text == ABSTAIN_MESSAGE
        yield Answer(text=text, citations=[] if abst else verify_citations(text, chunks), abstained=abst)


register("answerer", "provider", "raggity.answerer:ProviderAnswerer")
