from __future__ import annotations
import os
from typing import AsyncIterator

from openai import AsyncOpenAI

from .llm import LLMProvider
from .providers import is_local


class OpenAICompatProvider(LLMProvider):
    def __init__(self, model: str, base_url: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", temperature: float | None = None,
                 require_key: bool = False, ensure_provider: str | None = None,
                 external: bool = False) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        # backend="external": the server is owned by an external tool (Rigma);
        # raggity NEVER starts/stops it.  base_url is mandatory, and the auto-start
        # hook is forced off so the spawn path is structurally unreachable here —
        # not merely skipped.  Readiness is a lazy health probe instead (see _gate).
        self._external = external
        if external and not base_url:
            raise RuntimeError(
                "backend=external requires a base_url (the externally-managed "
                "OpenAI-compatible server URL); none is set. Set generation.base_url "
                "or run `rag model <name> -p external --base-url <url>`."
            )
        # When set, the named local runtime is ensured-running lazily on the first
        # real request (kept out of __init__ so construction is cheap/offline).
        # Forced None for external: no argument can re-enable auto-start.
        self._ensure_provider = None if external else ensure_provider
        self._ensured = False
        self._ready = False  # external readiness probe result, cached per instance
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
        if gen_cfg.backend == "external":
            # Never auto-start; never hard-require a key.  Local => keyless;
            # remote => use api_key_env only if the env var happens to be set
            # (the __init__ key lookup with require_key=False does exactly that).
            return cls(model=gen_cfg.model, base_url=base,
                       api_key_env=gen_cfg.api_key_env, temperature=gen_cfg.temperature,
                       require_key=False, ensure_provider=None, external=True)
        # A local server ignores the API key, so a missing key must NOT bite even
        # for backend="openai" (used by lmstudio/llamacpp/vllm/jan/koboldcpp).
        require_key = gen_cfg.backend == "openai" and not is_local(base)
        ensure_provider = _resolve_ensure_provider(gen_cfg, base)
        return cls(model=gen_cfg.model, base_url=base,
                   api_key_env=gen_cfg.api_key_env, temperature=gen_cfg.temperature,
                   require_key=require_key, ensure_provider=ensure_provider)

    def _gate(self) -> None:
        """Run the pre-request gate exactly once of the two mutually-exclusive
        paths: external => readiness probe (never spawns); otherwise => auto-start.
        External NEVER reaches ``_ensure``, so the spawn path is unreachable here."""
        if self._external:
            self._check_ready()
        else:
            self._ensure()

    def _check_ready(self) -> None:
        """Verify the externally-managed server is reachable (lazy, cached once).

        Probes health then /v1/models via :func:`providers.external_ready`.  Both
        dead => a clean RuntimeError naming the exact base_url and stating that
        backend=external never starts servers.  raggity does NOT launch anything."""
        if self._ready:
            return
        from . import providers  # noqa: PLC0415
        if not providers.external_ready(self.base_url):
            raise RuntimeError(
                f"external LLM server at {self.base_url} is unreachable - "
                f"start it (or let Rigma start it); raggity never launches servers "
                f"in backend=external mode"
            )
        self._ready = True

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
        self._gate()
        s = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            stream=True, **self._kw())
        async for chunk in s:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def complete(self, system: str, prompt: str) -> str:
        self._gate()
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
