# Packaging `rag` as a standalone Windows binary (PyInstaller)

This directory builds a self-contained `rag.exe` (onedir bundle) so raggity can
be distributed via winget / choco / scoop without the user installing Python.

**Status: VIABLE.** Verified end-to-end on Windows 11 / Python 3.12 with a
standalone `rag.exe` (repo venv NOT on PATH): `--help`, `init`, `ingest`,
`status`, `model --list`, `doctor`, and real grounded answers from both the
`ollama` and `claude` backends.

## Build

From the repo root, with the project venv active (or use its python directly):

```powershell
pip install pyinstaller
pyinstaller packaging/raggity.spec --noconfirm --clean `
    --distpath build-spike/dist --workpath build-spike/build
```

Output: `build-spike/dist/rag/rag.exe` plus its `_internal/` support tree.
Ship the whole `rag/` folder (zip it for portable/winget). `build-spike/` is
gitignored.

> The CI release pipeline (`.github/workflows/binaries.yml`) runs the spec with
> the default distpath, producing `dist/rag/rag[.exe]` on all three OSes.

Build time: ~50-65s. Sizes: **onedir ~460 MB, zipped ~173 MB.**
`rag.exe --help` cold start ~0.40s, warm ~0.34s.

## How it works / key decisions

### Entry point
`packaging/rag_entry.py` — the console script `rag = raggity.cli:app` points at a
Typer *object*, not a callable, so PyInstaller needs this thin shim that calls
`app()`.

### `claude_agent_sdk` bundled CLI — EXCLUDED (the big win)
The installed SDK ships a ~230 MB `claude.exe` under
`claude_agent_sdk/_bundled/`. Bundling it would nearly double the binary.

`SubprocessCLITransport._find_cli()` (in
`_internal/transport/subprocess_cli.py`) checks the bundled CLI **first**, then
falls back to `shutil.which("claude")` and a list of common install locations.
So when the bundled exe is absent, the SDK transparently uses a **system
`claude` on PATH**. The spec strips the `_bundled/` tree from the collected
bundle (belt-and-suspenders: it isn't collected by default anyway, as there is
no PyInstaller hook for the SDK).

Verified: with the bundle excluded, `rag ask` on the `claude` backend produced a
real grounded answer using the system `claude` at `~/.local/bin/claude`.

> **Distribution note:** the Claude backend requires Claude Code installed
> (`claude` on PATH). Local backends (ollama, and any OpenAI-compatible server:
> LM Studio, llama.cpp, vLLM, Jan, KoboldCpp) work out of the box with no extra
> install. `rag doctor` reports whether `claude` is reachable.

### Hidden imports (lazy / registry — invisible to static analysis)
raggity resolves backends through a dotted-path registry (`registry.py` →
`resolve()`) and uses inline `import` in many commands, so PyInstaller's static
analysis misses them. The spec lists them explicitly: `raggity.store` (lancedb),
`raggity.embedder`/`raggity.reranker` (fastembed), `raggity.answerer`,
`raggity.cached_embedder`, `raggity.llm`, `raggity.llm_openai` (ollama/openai),
`raggity.providers`, `raggity.doctor`, `raggity.conversation`,
`raggity.query_transform`, `raggity.graph`, `raggity.server`,
`raggity.connectors[.web/.github/.obsidian]`, plus `claude_agent_sdk`, `openai`,
`fastapi`, `uvicorn`.

Native/data-heavy packages are pulled with `collect_all`: `fastembed`,
`onnxruntime`, `lancedb`, `tokenizers`, `pyarrow` (DLLs + Rust extensions +
package data), plus `collect_submodules("rich")`.

### Data files
`src/raggity/web/index.html` (the `rag serve` web UI) is added as data at
`raggity/web/index.html` — `server.py` loads it via `Path(__file__).parent /
"web"`, which resolves correctly inside the bundle.

### Runtime model downloads — NOT bundled (by design)
fastembed downloads the embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB) and
reranker to the user cache **at runtime on first use**. These are intentionally
not bundled; the exe works against the standard fastembed/HF cache.

### OCR stack excluded to save ~110 MB
`readers.py` lazily imports `rapidocr_onnxruntime` + `pypdfium2` (which pull in
`cv2`, ~99 MB) — but only for the optional `raggity[ocr]` extra. PyInstaller
follows in-function imports, so these get collected unless excluded. The spec
excludes `cv2`, `rapidocr_onnxruntime`, `pypdfium2`: a base binary ships without
OCR (matching the base pip install, which also lacks OCR). To build an
OCR-capable exe, remove those three names from `excludes` in the spec.

## Verified commands (standalone exe, clean cwd, venv off PATH)

| command | result |
|---|---|
| `rag --help` | ok, ~0.34s warm |
| `rag init` | writes `raggity.toml` |
| `rag ingest` | `added=2` — fastembed + onnxruntime + lancedb all native |
| `rag status` | chunks/sources/model reported |
| `rag model --list` | lists local providers (ollama detected running) |
| `rag doctor` | all base checks ok; claude CLI + ollama detected |
| `rag model gemma3:4b -p ollama` + `rag ask ...` | real grounded answer + citation |
| `rag model ... -p claude` + `rag ask ...` | real grounded answer via system `claude` on PATH |

## Notes / caveats
- No raggity **source changes** were required. The exclude-the-bundled-CLI
  approach works purely via the spec.
- Console output shows an apostrophe as a stray byte (`�`) in some answers — a
  Windows console code-page display artifact, not a data problem.
- `qdrant` backend is not installed in this env; `raggity.qdrant_store` is
  intentionally omitted from hidden imports (default store is lancedb). Add it
  (and `qdrant_client`) if shipping qdrant support.
- onedir is recommended (fast startup, friendly for a winget/portable zip).
  onefile would repack the same payload into a single exe with a slower
  cold-start (self-extract each run); not needed here.
