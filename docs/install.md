# Installation

## Requirements

- Python **3.11+**
- A Claude subscription (`claude login`) **or** an `ANTHROPIC_API_KEY`

## 1. pip / pipx / uv (recommended for CLI use)

```bash
pip install raggity        # into the current environment
pipx install raggity       # isolated install, `rag` on PATH
uv tool install raggity    # isolated install, via uv
```

Both `rag` and `raggity` console scripts are registered — they are identical aliases.

---

## 2. Standalone binaries (no Python needed)

Prebuilt, self-contained builds are attached to every
[GitHub Release](https://github.com/IxMxAMAR/raggity/releases). No Python
installation is required.

| Platform | Asset | Install |
|---|---|---|
| Windows x64 | `raggity-<version>-windows-x86_64.zip` | Unzip anywhere, run `rag.exe`. Add the folder to `PATH` for a global `rag` command. |
| Linux x86_64 | `raggity-<version>-linux-x86_64.tar.gz` | `tar -xzf raggity-<version>-linux-x86_64.tar.gz && ./rag --help` |
| Linux (Debian/Ubuntu) | `raggity_<version>_amd64.deb` | `sudo apt install ./raggity_<version>_amd64.deb` — installs to `/opt/raggity` with a `/usr/bin/rag` symlink. |
| macOS (Apple Silicon) | `raggity-<version>-macos-arm64.tar.gz` | `tar -xzf raggity-<version>-macos-arm64.tar.gz && ./rag --help` |

!!! note "Backend requirements for the binaries"
    - The **Claude** backend needs Claude Code (`claude`) on your `PATH`.
    - **Local backends** (Ollama and any OpenAI-compatible server: LM Studio,
      llama.cpp, vLLM, Jan) work out of the box — nothing extra to install.
    - **OCR** for scanned PDFs and images is **pip-only**: the binaries omit the
      OCR stack; install `raggity[ocr]` into a Python environment if you need it.

The embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB) is downloaded to your
cache on first use, exactly like the pip install.

---

## 3. Package managers

```bash
# Scoop (Windows)
scoop bucket add raggity https://github.com/IxMxAMAR/scoop-raggity
scoop install raggity

# Homebrew (macOS / Linux) — binary formula
brew tap ixmxamar/raggity
brew install raggity
```

winget (`winget install IxMxAMAR.raggity`) and Chocolatey (`choco install raggity`)
are **submission in progress**.

---

## Optional extras

Install extras for additional capabilities:

| Extra | Install | Adds |
|---|---|---|
| `docs` | `pip install raggity[docs]` | Back-compat alias — `.docx`, `.html`, `.pptx` are in the base install |
| `all` | `pip install raggity[all]` | All optional extras except `docs-site` (server, qdrant, ocr, openai, web, graph, otel, watch) |
| `ocr` | `pip install raggity[ocr]` | RapidOCR + pypdfium2 for scanned PDFs and images |
| `server` | `pip install raggity[server]` | FastAPI HTTP server + web chat UI |
| `qdrant` | `pip install raggity[qdrant]` | Qdrant vector store backend |
| `watch` | `pip install raggity[watch]` | `rag watch` filesystem watcher daemon |
| `openai` | `pip install raggity[openai]` | OpenAI-compatible API backend (also needed for Ollama) |
| `web` | `pip install raggity[web]` | `rag ingest-url` web crawling connector |
| `graph` | `pip install raggity[graph]` | GraphRAG knowledge-graph support |
| `otel` | `pip install raggity[otel]` | OpenTelemetry tracing + metrics export |
| `docs-site` | `pip install raggity[docs-site]` | MkDocs Material (for building this documentation site) |

Common document formats (`.md`, `.txt`, `.pdf`, `.docx`, `.html`, `.pptx`, `.csv`) work out of the box with `pip install raggity`. Only image OCR and scanned PDFs need `raggity[ocr]`.

You can combine extras:

```bash
pip install raggity[docs,ocr,server,qdrant]
```

---

## Auth setup

raggity answers questions via the **Claude Agent SDK**. Choose one of:

### Subscription (recommended)

Log in once with the Claude CLI:

```bash
claude login
```

Leave `generation.auth = "auto"` in your config (the default). raggity uses your subscription session when no `ANTHROPIC_API_KEY` is set — no API key needed.

### API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Set `auth = "api_key"` in `raggity.toml` under `[generation]` if you want to force key-only mode.

### Auth modes

| Value | Behaviour |
|---|---|
| `"auto"` (default) | Uses key if `ANTHROPIC_API_KEY` is set, otherwise falls back to subscription session |
| `"subscription"` | Always uses `claude login` session; `ANTHROPIC_API_KEY` is **ignored** |
| `"api_key"` | Requires `ANTHROPIC_API_KEY`; raises at startup if missing |

---

## Platform support

| Platform | Status |
|---|---|
| Linux x86_64 | Supported |
| macOS Apple Silicon (ARM) | Supported |
| Windows x64 | Supported |
| Intel macOS x86_64 | Partial — onnxruntime wheels may be missing for some Python versions |
| Windows on ARM | Partial — onnxruntime and LanceDB binary wheels not consistently available |

If you hit a missing wheel on an unsupported platform, open an issue or try building from source. The CPU-only path is the most portable.
