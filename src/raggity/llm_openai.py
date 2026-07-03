from __future__ import annotations
import os
from typing import AsyncIterator

from openai import AsyncOpenAI

from .llm import LLMProvider
from .providers import is_local


class OpenAICompatProvider(LLMProvider):
    def __init__(self, model: str, base_url: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", temperature: float | None = None,
                 require_key: bool = False, ensure_provider: str | None = None) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        # When set, the named local runtime is ensured-running lazily on the first
        # real request (kept out of __init__ so construction is cheap/offline).
        self._ensure_provider = ensure_provider
        self._ensured = False
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
        # A local server ignores the API key, so a missing key must NOT bite even
        # for backend="openai" (used by lmstudio/llamacpp/vllm/jan/koboldcpp).
        require_key = gen_cfg.backend == "openai" and not is_local(base)
        ensure_provider = _resolve_ensure_provider(gen_cfg, base)
        return cls(model=gen_cfg.model, base_url=base,
                   api_key_env=gen_cfg.api_key_env, temperature=gen_cfg.temperature,
                   require_key=require_key, ensure_provider=ensure_provider)

    def _ensure(self) -> None:
        """Auto-start the local runtime once per instance (best-effort, cached)."""
        if self._ensured:
            return
        self._ensured = True
        if not self._ensure_provider:
            return
        try:
            from . import providers  # noqa: PLC0415
            providers.ensure_running(self._ensure_provider, self.base_url)
        except Exception:
            pass

    def _kw(self) -> dict:
        return {} if self.temperature is None else {"temperature": self.temperature}

    async def stream(self, system: str, prompt: str):
        self._ensure()
        s = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            stream=True, **self._kw())
        async for chunk in s:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def complete(self, system: str, prompt: str) -> str:
        self._ensure()
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


def _resolve_ensure_provider(gen_cfg, base: str | None) -> str | None:
    """Pick the local runtime (if any) to auto-start lazily for this config.

    ollama is the must-have.  For backend="openai" only a trivially-identifiable
    local default (base_url matches a known provider AND its binary is present)
    is auto-started; anything ambiguous is left alone.
    """
    if not getattr(gen_cfg, "auto_start", True):
        return None
    from . import providers  # noqa: PLC0415
    if gen_cfg.backend == "ollama" and is_local(base or providers.default_base_url("ollama")):
        return "ollama"
    if gen_cfg.backend == "openai" and is_local(base):
        cand = providers.provider_for_base_url(base)
        if cand and providers.auto_startable(cand):
            return cand
    return None
