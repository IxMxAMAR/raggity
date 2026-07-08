from __future__ import annotations
import logging
import os
from abc import ABC, abstractmethod
from typing import AsyncIterator

# Kill claude_agent_sdk's per-call `claude -v` Node subprocess spawn (it reads this
# from the parent env, not options.env, and runs regardless of caching). Must be set
# before the SDK is ever imported, so it lives at module import time here.
os.environ.setdefault("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")

log = logging.getLogger("raggity.llm")

# Env vars that carry Anthropic credentials; all are stripped in subscription mode.
_ANTHROPIC_CRED_VARS = {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"}

# Lazy SDK bindings: claude_agent_sdk's __init__ eagerly imports the full mcp package
# (incl. server stack), costing ~344ms on every `raggity.llm` import even when no LLM
# call is ever made (e.g. `rag status`, `rag --help`). Deferred via _ensure_sdk() below.
# Per-name None guards keep this import-order-independent AND test-patch-safe: the 53
# `monkeypatch.setattr(llm, "query", ...)` / 52 `AssistantMessage` sites across the
# test suite set these to non-None fakes, and _ensure_sdk() only fills names that are
# still None, so it never clobbers a test-injected fake.
query = None
ClaudeAgentOptions = None
AssistantMessage = None
# In-process SDK MCP tool support (used by agentic.py). Same guarded-globals
# discipline as the symbols above: _ensure_sdk() only fills them when still None,
# so a test that patches them stays intact.
tool = None
create_sdk_mcp_server = None
# StreamEvent carries the SDK's incremental partial-message deltas. It is
# feature-detected (older SDKs lack it): when unavailable it stays None and the
# provider falls back to yielding the final AssistantMessage snapshot only —
# i.e. exactly today's behavior. Its own resolve flag avoids re-importing the
# SDK on every call in the (rare) case the class is genuinely absent.
StreamEvent = None
_stream_event_checked = False


def _ensure_sdk() -> None:
    """Import claude_agent_sdk on first real use, filling in whichever of the
    module globals above are still unset (None). No-op if all are already bound
    (either by a prior real call, or by a test's monkeypatch)."""
    global query, ClaudeAgentOptions, AssistantMessage, StreamEvent, _stream_event_checked
    global tool, create_sdk_mcp_server
    if query is None:
        from claude_agent_sdk import query as _query
        query = _query
    if ClaudeAgentOptions is None:
        from claude_agent_sdk import ClaudeAgentOptions as _ClaudeAgentOptions
        ClaudeAgentOptions = _ClaudeAgentOptions
    if AssistantMessage is None:
        from claude_agent_sdk import AssistantMessage as _AssistantMessage
        AssistantMessage = _AssistantMessage
    if tool is None:
        from claude_agent_sdk import tool as _tool
        tool = _tool
    if create_sdk_mcp_server is None:
        from claude_agent_sdk import create_sdk_mcp_server as _create_sdk_mcp_server
        create_sdk_mcp_server = _create_sdk_mcp_server
    if StreamEvent is None and not _stream_event_checked:
        _stream_event_checked = True
        try:
            from claude_agent_sdk import StreamEvent as _StreamEvent
            StreamEvent = _StreamEvent
        except Exception:
            StreamEvent = None


def base_options_kwargs(system: str, model: str, auth: str) -> dict:
    """Shared ClaudeAgentOptions kwargs for any Claude call (single-shot or agentic).

    Applies the subscription credential-strip and the api_key guard EXACTLY as
    ``ClaudeProvider._options`` does, and sets the isolation defaults
    (``permission_mode="dontAsk"``, ``setting_sources=[]``).  Callers add their
    own ``allowed_tools`` / ``max_turns`` / ``mcp_servers`` on top.  The env logic
    runs before any SDK import, so an api_key-without-key error surfaces without
    touching claude_agent_sdk.
    """
    env = None
    if auth == "subscription":
        env = {k: v for k, v in os.environ.items()
               if k not in _ANTHROPIC_CRED_VARS}
    elif auth == "api_key" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("auth='api_key' but ANTHROPIC_API_KEY is not set. "
                           "Set the key or use auth='subscription' after `claude login`.")
    kw = dict(system_prompt=system, model=model,
              permission_mode="dontAsk", setting_sources=[])
    if env is not None:
        kw["env"] = env
    return kw


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
        kw = base_options_kwargs(system, self.model, self.auth)
        _ensure_sdk()
        kw.update(allowed_tools=[], max_turns=1, include_partial_messages=True)
        return ClaudeAgentOptions(**kw)

    async def stream(self, system: str, prompt: str):
        opts = self._options(system)
        streamed = False  # whether any incremental text delta was yielded this turn
        async for message in query(prompt=prompt, options=opts):
            if StreamEvent is not None and isinstance(message, StreamEvent):
                # Real token streaming: the SDK forwards raw Anthropic API stream
                # events. Text arrives as content_block_delta / text_delta.
                ev = getattr(message, "event", None) or {}
                if ev.get("type") == "content_block_delta":
                    delta = ev.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        t = delta.get("text")
                        if t:
                            streamed = True
                            yield t
            elif isinstance(message, AssistantMessage):
                # The final AssistantMessage is one complete snapshot. If we already
                # streamed the text via deltas this turn, suppress it to avoid
                # duplication; otherwise (no partial events — e.g. every existing
                # mock) yield it, preserving today's behavior.
                if streamed:
                    continue
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
    if backend in ("openai", "ollama", "external"):
        from .llm_openai import OpenAICompatProvider  # added in Task A4
        return OpenAICompatProvider.from_config(gen_cfg)
    raise ValueError(
        f"unknown generation.backend {backend!r} (expected claude|openai|ollama|external)")
