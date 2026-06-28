# Installation

## Requirements

- Python **3.11+**
- A Claude subscription (`claude login`) **or** an `ANTHROPIC_API_KEY`

## Core install

```bash
pip install raggity
```

Both `rag` and `raggity` console scripts are registered — they are identical aliases.

---

## Optional extras

Install extras for additional capabilities:

| Extra | Install | Adds |
|---|---|---|
| `docs` | `pip install raggity[docs]` | `.docx`, `.html`, `.pptx` reader support |
| `ocr` | `pip install raggity[ocr]` | RapidOCR + pypdfium2 for scanned PDFs and images |
| `server` | `pip install raggity[server]` | FastAPI HTTP server + web chat UI |
| `qdrant` | `pip install raggity[qdrant]` | Qdrant vector store backend |
| `watch` | `pip install raggity[watch]` | `rag watch` filesystem watcher daemon |
| `openai` | `pip install raggity[openai]` | OpenAI-compatible API backend (also needed for Ollama) |
| `web` | `pip install raggity[web]` | `rag ingest-url` web crawling connector |
| `graph` | `pip install raggity[graph]` | GraphRAG knowledge-graph support |
| `otel` | `pip install raggity[otel]` | OpenTelemetry tracing + metrics export |
| `docs-site` | `pip install raggity[docs-site]` | MkDocs Material (for building this documentation site) |

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
