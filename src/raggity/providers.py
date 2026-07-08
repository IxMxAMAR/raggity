"""Discovery + auto-start for local LLM runtimes (stdlib-only).

No network dependencies: probes use ``urllib`` with short timeouts, process
discovery uses ``shutil.which`` + well-known install paths, and auto-start
spawns a detached child via ``subprocess``.  Every probe is best-effort: any
exception means "not running" / "not installed", never a crash.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlparse

_PROBE_TIMEOUT = 1.5  # seconds; local servers respond fast or not at all
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass
class ProviderStatus:
    name: str
    base_url: str
    running: bool
    installed: bool
    models: list[str] = field(default_factory=list)
    auto_startable: bool = False


# Registry of known local runtimes.  ``native`` marks Ollama's non-OpenAI probe
# endpoints (/api/version + /api/tags).  ``auto_start`` is the spawn argv with
# ``{exe}`` substituted by the discovered binary; None means discovery-only.
_SPECS: dict[str, dict] = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "native": True,
        "exe_names": ["ollama"],
        "exe_paths": [os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")],
        "auto_start": ["{exe}", "serve"],
        "start_needs_exe": True,
        "hint": "install from https://ollama.com then `ollama pull <model>`",
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "native": False,
        "exe_names": ["lms"],
        "exe_paths": [os.path.expandvars(r"%LOCALAPPDATA%\Programs\LM Studio")],
        "auto_start": ["{exe}", "server", "start"],
        "start_needs_exe": True,
        "hint": "start the LM Studio app (or install the `lms` CLI to auto-start)",
    },
    "llamacpp": {
        "base_url": "http://localhost:8080/v1",
        "native": False,
        "exe_names": ["llama-server"],
        "exe_paths": [],
        "auto_start": None,
        "start_needs_exe": True,
        "hint": "run `llama-server -m <model.gguf> --port 8080`",
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "native": False,
        "exe_names": ["vllm"],
        "exe_paths": [],
        "auto_start": None,
        "start_needs_exe": True,
        "hint": "run `vllm serve <model>`",
    },
    "jan": {
        "base_url": "http://localhost:1337/v1",
        "native": False,
        "exe_names": ["jan"],
        "exe_paths": [],
        "auto_start": None,
        "start_needs_exe": True,
        "hint": "start the Jan app and enable its local API server",
    },
    "koboldcpp": {
        "base_url": "http://localhost:5001/v1",
        "native": False,
        "exe_names": ["koboldcpp"],
        "exe_paths": [],
        "auto_start": None,
        "start_needs_exe": True,
        "hint": "run koboldcpp with `--port 5001`",
    },
}

# CLI provider alias -> (backend, base_url override or None).  Local OpenAI-compat
# runtimes map to backend "openai" with their default base_url; ollama keeps its
# native backend.
BACKEND_ALIASES: dict[str, tuple[str, str | None]] = {
    "claude": ("claude", None),
    "anthropic": ("claude", None),
    "openai": ("openai", None),
    "ollama": ("ollama", None),
    # Externally-managed OpenAI-compatible server (Rigma owns lifecycle).  No
    # default base_url — it MUST be supplied (config or `--base-url`).
    "external": ("external", None),
    "lmstudio": ("openai", _SPECS["lmstudio"]["base_url"]),
    "llamacpp": ("openai", _SPECS["llamacpp"]["base_url"]),
    "vllm": ("openai", _SPECS["vllm"]["base_url"]),
    "jan": ("openai", _SPECS["jan"]["base_url"]),
    "koboldcpp": ("openai", _SPECS["koboldcpp"]["base_url"]),
}


def is_local(base_url: str | None) -> bool:
    """True when *base_url* points at a loopback host."""
    if not base_url:
        return False
    try:
        host = urlparse(base_url).hostname or ""
    except Exception:
        return False
    return host in _LOCAL_HOSTS


def _root(base_url: str) -> str:
    """Return the server root by stripping a trailing ``/v1``."""
    b = base_url.rstrip("/")
    if b.endswith("/v1"):
        b = b[:-3]
    return b.rstrip("/")


def _get_json(url: str, timeout: float = _PROBE_TIMEOUT) -> dict:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8", "replace"))


def _exe(spec: dict) -> str | None:
    """Return a runnable binary path for *spec*, or None.

    A directory in ``exe_paths`` (e.g. an app install folder) is NOT runnable and
    does not count here; it only signals "installed".
    """
    for name in spec.get("exe_names", []):
        found = shutil.which(name)
        if found:
            return found
    for path in spec.get("exe_paths", []):
        if path and os.path.isfile(path):
            return path
    return None


def _installed(spec: dict) -> bool:
    if _exe(spec) is not None:
        return True
    return any(p and os.path.exists(p) for p in spec.get("exe_paths", []))


def auto_startable(name: str) -> bool:
    """True when *name* can be auto-started right now (no network probe)."""
    spec = _SPECS.get(name)
    if not spec or not spec.get("auto_start"):
        return False
    if spec.get("start_needs_exe"):
        return _exe(spec) is not None
    return True


def default_base_url(name: str) -> str | None:
    spec = _SPECS.get(name)
    return spec["base_url"] if spec else None


def provider_for_base_url(base_url: str | None) -> str | None:
    """Return the registry name whose default base_url matches *base_url*."""
    if not base_url:
        return None
    norm = base_url.rstrip("/")
    for name, spec in _SPECS.items():
        if spec["base_url"].rstrip("/") == norm:
            return name
    return None


def external_ready(base_url: str | None, timeout: float = 2.0) -> bool:
    """Readiness probe for ``backend="external"`` (Rigma-managed server).

    GET ``<root>/health``; on ANY failure fall back to GET ``<root>/v1/models``.
    Returns True on the first success, False when both fail.  This NEVER starts a
    server — ``backend=external`` means the runtime's lifecycle is owned elsewhere
    (Rigma).  ``<root>`` is *base_url* with a trailing ``/v1`` stripped, so both
    ``http://host:9999`` and ``http://host:9999/v1`` probe the same endpoints.
    """
    if not base_url:
        return False
    root = _root(base_url)
    try:
        _get_json(root + "/health", timeout=timeout)
        return True
    except Exception:
        pass
    try:
        _get_json(root + "/v1/models", timeout=timeout)
        return True
    except Exception:
        return False


def _reachable(name: str, base_url: str) -> bool:
    """Cheap liveness probe (no model listing)."""
    spec = _SPECS.get(name)
    if spec is None:
        return False
    try:
        if spec["native"]:
            _get_json(_root(base_url) + "/api/version")
        else:
            _get_json(base_url.rstrip("/") + "/models")
        return True
    except Exception:
        return False


def _models(name: str, base_url: str) -> list[str]:
    spec = _SPECS.get(name)
    if spec is None:
        return []
    try:
        if spec["native"]:
            data = _get_json(_root(base_url) + "/api/tags")
            return [m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)]
        data = _get_json(base_url.rstrip("/") + "/models")
        return [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]
    except Exception:
        return []


def _status(name: str) -> ProviderStatus:
    spec = _SPECS[name]
    base = spec["base_url"]
    running = _reachable(name, base)
    models = _models(name, base) if running else []
    return ProviderStatus(
        name=name, base_url=base, running=running,
        installed=_installed(spec), models=models,
        auto_startable=auto_startable(name),
    )


def discover() -> list[ProviderStatus]:
    """Probe every known local runtime; returns a status row per provider."""
    return [_status(name) for name in _SPECS]


def _spawn_detached(cmd: list[str]) -> None:
    kwargs: dict = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL)
    if os.name == "nt":
        flags = 0
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(cmd, **kwargs)  # noqa: S603


def ensure_running(name: str, base_url: str | None = None, timeout: float = 10.0) -> bool:
    """Ensure *name* is reachable, auto-starting it when possible.

    Returns True if the server is (or becomes) reachable.  A no-op fast path when
    already up.  When down and a runnable binary exists, spawns a detached server
    and polls the liveness endpoint (0.5s interval) up to *timeout* seconds.
    """
    spec = _SPECS.get(name)
    if spec is None:
        return False
    base = base_url or spec["base_url"]
    if _reachable(name, base):
        return True
    if not spec.get("auto_start"):
        return False
    exe = _exe(spec)
    if spec.get("start_needs_exe") and exe is None:
        return False
    cmd = [exe if tok == "{exe}" else tok for tok in spec["auto_start"]]
    try:
        _spawn_detached(cmd)
    except Exception:
        return False
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _reachable(name, base):
            return True
        time.sleep(0.5)
    return _reachable(name, base)
