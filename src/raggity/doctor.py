"""`rag doctor` diagnostics.

Each check is a small, individually-testable helper returning
``(status, detail, hint)`` where ``status`` is one of ``ok`` / ``warn`` / ``FAIL``
/ ``info`` and ``hint`` is a one-line remedy shown for warn/fail rows.
``run_doctor`` calls the helpers via this module's namespace so tests can
monkeypatch any single check.
"""
from __future__ import annotations

import glob
import os
import shutil
import tempfile

OK = "ok"
WARN = "warn"
FAIL = "FAIL"
INFO = "info"

# Optional extras -> an importable module that proves the extra is installed.
_EXTRAS: dict[str, str] = {
    "ocr": "rapidocr_onnxruntime",
    "web": "trafilatura",
    "qdrant": "qdrant_client",
    "server": "fastapi",
    "watch": "watchdog",
    "graph": "networkx",
    "otel": "opentelemetry",
}


def check_version() -> tuple[str, str, str]:
    from . import __version__
    return (OK, f"raggity {__version__}", "")


def check_config(cfg_path) -> tuple[str, str, str]:
    if cfg_path is None:
        return (WARN, "no config file found", "run `rag init` to create raggity.toml")
    try:
        import tomllib
        with open(cfg_path, "rb") as fh:
            tomllib.load(fh)
    except Exception as exc:
        return (FAIL, f"{cfg_path} does not parse", f"fix the TOML syntax: {exc}")
    return (OK, str(cfg_path), "")


def check_sources(cfg) -> tuple[str, str, str]:
    patterns = list(cfg.sources.include)
    if not patterns:
        return (WARN, "no include patterns", "add [sources] include globs then `rag ingest`")
    total = 0
    for pat in patterns:
        expanded = os.path.expanduser(pat)
        try:
            total += sum(1 for p in glob.iglob(expanded, recursive=True) if os.path.isfile(p))
        except Exception:
            continue
        if total > 10_000:
            break
    if total == 0:
        return (WARN, "0 files match include patterns", "check [sources] include globs")
    if total > 10_000:
        return (WARN, f"{total}+ files match", "narrow [sources] include/exclude to avoid huge scans")
    return (OK, f"{total} file(s) match include patterns", "")


def check_index(cfg) -> tuple[str, str, str]:
    from .core import Raggity
    try:
        rag = Raggity(cfg)
        chunks = rag.store.count()
        sources = len(rag.store.all_source_paths())
    except Exception as exc:
        return (WARN, "index could not be opened", f"reindex may be needed: {exc}")
    if chunks == 0:
        return (OK, "empty index", "run `rag ingest` to populate it")
    return (OK, f"{chunks} chunk(s), {sources} source(s)", "")


def check_embedding(cfg) -> tuple[str, str, str]:
    from .embedder import FastEmbedEmbedder
    try:
        emb = FastEmbedEmbedder(
            model_name=cfg.embedding.model,
            provider=cfg.embedding.provider,
            batch_size=cfg.embedding.batch_size,
            parallel=cfg.embedding.parallel,
        )
        dim = emb.dim
    except Exception as exc:
        return (FAIL, f"embedding model {cfg.embedding.model} failed to load",
                f"check the model name / provider / install: {exc}")
    return (OK, f"{cfg.embedding.model} dim={dim} provider={cfg.embedding.provider}", "")


def check_generation(cfg) -> tuple[str, str, str]:
    from . import providers
    gen = cfg.generation
    backend = gen.backend
    if backend == "claude":
        try:
            import claude_agent_sdk  # noqa: F401
        except Exception:
            return (FAIL, "backend=claude but claude_agent_sdk not importable",
                    "pip install claude-agent-sdk")
        if os.environ.get("ANTHROPIC_API_KEY") or shutil.which("claude"):
            return (OK, "backend=claude (subscription/CLI available)", "")
        return (WARN, "backend=claude but no auth signal",
                "no API key and no claude CLI found - run `claude login` or set ANTHROPIC_API_KEY")
    if backend == "ollama":
        base = gen.base_url or providers.default_base_url("ollama")
        started = providers.ensure_running("ollama", base) if gen.auto_start else providers._reachable("ollama", base)
        if not started:
            return (FAIL, f"backend=ollama unreachable at {base}", "is Ollama running?")
        models = providers._models("ollama", base)
        detail = f"backend=ollama reachable at {base}"
        if models and gen.model not in models:
            return (WARN, f"{detail} but model {gen.model!r} not pulled",
                    f"model not pulled? run `ollama pull {gen.model}`")
        return (OK, f"{detail} model={gen.model}", "")
    if backend == "external":
        base = gen.base_url
        if not base:
            return (FAIL, "backend=external but no base_url set",
                    "set generation.base_url to the external server URL "
                    "(or `rag model <name> -p external --base-url <url>`)")
        # Same readiness probe the provider uses; NEVER auto-starts (Rigma owns it).
        if providers.external_ready(base):
            return (OK, f"external server reachable at {base}", "")
        return (FAIL, f"external server unreachable at {base} - backend=external never auto-starts",
                "start the server (Rigma owns its lifecycle); raggity never launches it")
    if backend == "openai":
        base = gen.base_url or "https://api.openai.com/v1"
        key = os.environ.get(gen.api_key_env)
        if not key and not providers.is_local(base):
            return (WARN, f"backend=openai but {gen.api_key_env} not set",
                    f"set {gen.api_key_env}")
        try:
            providers._get_json(base.rstrip("/") + "/models")
        except Exception:
            return (WARN, f"backend=openai endpoint {base} not reachable",
                    "network issue or invalid key (best-effort check)")
        return (OK, f"backend=openai model={gen.model} at {base}", "")
    return (WARN, f"backend={backend!r} unrecognized", "expected claude|openai|ollama|external")


def check_profile(cfg) -> list[tuple[str, str, str, str]]:
    """Report the active config `profile` preset, if any (empty list = unset)."""
    if cfg.profile == "low-ram":
        return [(
            "profile", INFO,
            "low-ram (rerank/graph/caches off, embedded lancedb)", "",
        )]
    return []


def check_index_writable(cfg) -> tuple[str, str, str]:
    path = cfg.index.path
    try:
        os.makedirs(path, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path, prefix=".doctor_")
        os.close(fd)
        os.remove(tmp)
    except Exception as exc:
        return (FAIL, f"index dir not writable: {path}", f"check permissions: {exc}")
    return (OK, f"index dir writable: {path}", "")


def check_extras() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for extra, module in _EXTRAS.items():
        try:
            __import__(module)
            rows.append((f"extra:{extra}", OK, "installed", ""))
        except Exception:
            rows.append((f"extra:{extra}", INFO, f"not installed (pip install raggity[{extra}])", ""))
    return rows


def check_providers() -> list[tuple[str, str, str, str]]:
    from . import providers
    rows: list[tuple[str, str, str, str]] = []
    for st in providers.discover():
        if st.running:
            models = f" [{len(st.models)} model(s)]" if st.models else ""
            detail = f"running at {st.base_url}{models}"
            status = OK
        elif st.installed:
            start = " (auto-startable)" if st.auto_startable else ""
            detail = f"installed, not running{start}"
            status = INFO
        else:
            detail = "not found"
            status = INFO
        rows.append((f"provider:{st.name}", status, detail, ""))
    return rows


# ASCII markers (Windows-console-safe) + their Rich style colors.
_MARKER = {OK: "[ok]", WARN: "[warn]", FAIL: "[FAIL]", INFO: "[--]"}
_STYLE = {OK: "green", WARN: "yellow", FAIL: "red", INFO: "dim"}


def run_doctor(config, console) -> int:
    """Run all checks, print a marker checklist, return an exit code (0 ok, 1 any FAIL).

    Rows are printed via :class:`rich.text.Text` (append with explicit styles) so
    literal ``[ok]``/``[FAIL]`` markers and bracketed details never collide with
    Rich markup or emoji shortcode substitution.
    """
    from rich.text import Text
    from .config import _find_config_path, load_config

    cfg_path = _find_config_path(config)
    cfg = load_config(config)

    rows: list[tuple[str, str, str, str]] = []
    rows.append(("version", *check_version()))
    rows.append(("config", *check_config(cfg_path)))
    rows.extend(check_profile(cfg))
    rows.append(("sources", *check_sources(cfg)))
    rows.append(("index", *check_index(cfg)))
    rows.append(("embedding", *check_embedding(cfg)))
    rows.append(("generation", *check_generation(cfg)))
    rows.append(("index-writable", *check_index_writable(cfg)))
    rows.extend(check_extras())
    rows.extend(check_providers())

    any_fail = False
    for label, status, detail, hint in rows:
        if status == FAIL:
            any_fail = True
        line = Text()
        line.append(_MARKER.get(status, "[--]"), style=_STYLE.get(status, "dim"))
        line.append(f" {label}: {detail}")
        console.print(line)
        if status in (WARN, FAIL) and hint:
            console.print(Text(f"       hint: {hint}", style="dim"))
    return 1 if any_fail else 0
