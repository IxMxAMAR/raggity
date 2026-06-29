from __future__ import annotations
import os
from typing import AsyncIterator

from openai import AsyncOpenAI

from .llm import LLMProvider


class OpenAICompatProvider(LLMProvider):
    def __init__(self, model: str, base_url: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", temperature: float | None = None,
                 require_key: bool = False) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        key = os.environ.get(api_key_env)
        if not key:
            if require_key:
                raise RuntimeError(
                    f"{api_key_env} is not set. "
                    f"Set the environment variable or use a different backend."
                )
            key = "not-needed"  # local/ollama servers ignore the key
        self._client = AsyncOpenAI(base_url=base_url, api_key=key)

    @classmethod
    def from_config(cls, gen_cfg) -> "OpenAICompatProvider":
        base = gen_cfg.base_url
        if gen_cfg.backend == "ollama" and not base:
            base = "http://localhost:11434/v1"
        require_key = gen_cfg.backend == "openai"
        return cls(model=gen_cfg.model, base_url=base,
                   api_key_env=gen_cfg.api_key_env, temperature=gen_cfg.temperature,
                   require_key=require_key)

    def _kw(self) -> dict:
        return {} if self.temperature is None else {"temperature": self.temperature}

    async def stream(self, system: str, prompt: str):
        s = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            stream=True, **self._kw())
        async for chunk in s:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def complete(self, system: str, prompt: str) -> str:
        r = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            **self._kw())
        return (r.choices[0].message.content or "").strip()

    async def aclose(self) -> None:
        """Close the underlying httpx session.  Safe to call multiple times."""
        try:
            await self._client.close()
        except Exception:
            pass
