from __future__ import annotations
import logging
import os
from abc import ABC, abstractmethod
from typing import AsyncIterator

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage

log = logging.getLogger("raggity.llm")

# Env vars that carry Anthropic credentials; all are stripped in subscription mode.
_ANTHROPIC_CRED_VARS = {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"}


class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, system: str, prompt: str) -> str: ...
    @abstractmethod
    def stream(self, system: str, prompt: str) -> AsyncIterator[str]: ...

    async def aclose(self) -> None:
        """Release any underlying resources.  Default implementation is a no-op."""


class ClaudeProvider(LLMProvider):
    def __init__(self, model: str = "claude-opus-4-8", auth: str = "auto") -> None:
        self.model = model
        self.auth = auth

    def _options(self, system: str) -> ClaudeAgentOptions:
        env = None
        if self.auth == "subscription":
            env = {k: v for k, v in os.environ.items()
                   if k not in _ANTHROPIC_CRED_VARS}
        elif self.auth == "api_key" and not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("auth='api_key' but ANTHROPIC_API_KEY is not set. "
                               "Set the key or use auth='subscription' after `claude login`.")
        kw = dict(system_prompt=system, model=self.model,
                  allowed_tools=[], permission_mode="dontAsk")
        if env is not None:
            kw["env"] = env
        return ClaudeAgentOptions(**kw)

    async def stream(self, system: str, prompt: str):
        opts = self._options(system)
        try:
            opts.include_partial_messages = True
        except Exception as exc:
            log.debug("include_partial_messages not supported by SDK version: %s", exc)
        async for message in query(prompt=prompt, options=opts):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    t = getattr(block, "text", None)
                    if t:
                        yield t

    async def complete(self, system: str, prompt: str) -> str:
        parts: list[str] = []
        async for t in self.stream(system, prompt):
            parts.append(t)
        return "".join(parts).strip()

    async def aclose(self) -> None:
        """No-op: ClaudeProvider holds no persistent connection."""


def build_provider(gen_cfg) -> LLMProvider:
    backend = gen_cfg.backend
    if backend == "claude":
        return ClaudeProvider(model=gen_cfg.model, auth=gen_cfg.auth)
    if backend in ("openai", "ollama"):
        from .llm_openai import OpenAICompatProvider  # added in Task A4
        return OpenAICompatProvider.from_config(gen_cfg)
    raise ValueError(f"unknown generation.backend {backend!r} (expected claude|openai|ollama)")
