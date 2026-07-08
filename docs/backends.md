# Backends

raggity supports four **LLM generation backends** and two **vector store backends**.

---

## LLM backends

Controlled by `generation.backend` in `raggity.toml`.

### Claude (default)

Uses the **Claude Agent SDK**. No extra dependencies. Requires a Claude subscription or API key.

```toml
[generation]
backend = "claude"
model = "claude-opus-4-8"
auth = "auto"
```

**Auth modes:**

| `auth` value | Behaviour |
|---|---|
| `"auto"` (default) | Uses `ANTHROPIC_API_KEY` if set, otherwise falls back to `claude login` subscription session |
| `"subscription"` | Always uses `claude login` session; `ANTHROPIC_API_KEY` is **ignored** |
| `"api_key"` | Requires `ANTHROPIC_API_KEY`; raises at startup if missing |

Set up once with the Claude CLI:

```bash
claude login
```

### OpenAI-compatible

Any OpenAI-compatible API endpoint — OpenAI, Azure OpenAI, Together, Groq, etc. Requires the `openai` extra:

```bash
pip install raggity[openai]
```

```toml
[generation]
backend = "openai"
model = "gpt-4o-mini"
base_url = "https://api.openai.com/v1"   # default; any OpenAI-compatible URL works
api_key_env = "OPENAI_API_KEY"           # env var name holding the API key
```

Set the API key:

```bash
export OPENAI_API_KEY=sk-...
```

The `auth` field is ignored for this backend.

### Ollama (offline)

Runs against a local [Ollama](https://ollama.com) server — no API key required. Reuses the `openai` extra (OpenAI client):

```bash
pip install raggity[openai]
ollama pull llama3.1
```

```toml
[generation]
backend = "ollama"
model = "llama3.1"
# base_url defaults to http://localhost:11434/v1 — omit unless Ollama is on a different port
```

The `auth` field is ignored for this backend.

### External (managed by another tool)

Targets an OpenAI-compatible server whose lifecycle is owned outside raggity —
for example [Rigma](https://github.com/IxMxAMAR/rigma), or a server you start
and manage yourself. raggity **never** starts, stops, or otherwise manages
this server's process.

```toml
[generation]
backend = "external"
model = "some-model"
base_url = "http://127.0.0.1:9999/v1"   # required — no default
```

`base_url` is required — raggity raises at startup if it's missing. Readiness
is a lazy probe (`GET <root>/health`, falling back to `GET <root>/v1/models`)
rather than an auto-start attempt; if both probes fail, the error names the
exact `base_url` and states that `backend=external` never launches servers.

Switch to it from the CLI:

```bash
rag model some-model -p external --base-url http://127.0.0.1:9999
```

The `auth` field is ignored for this backend.

---

## Vector store backends

Controlled by `index.backend` in `raggity.toml`.

### LanceDB (default)

LanceDB is the default and requires **no extra install**. Data is stored locally in the directory specified by `index.path`.

```toml
[index]
backend = "lancedb"
path = ".raggity/index"   # relative to cwd (default)
```

**Recommended for:** single-user local deployments.

### Qdrant

Qdrant is recommended for large-scale or multi-user deployments. Install the extra:

```bash
pip install raggity[qdrant]
```

Configure in `raggity.toml`:

```toml
[index]
backend = "qdrant"
qdrant_location = "http://localhost:6333"   # remote Qdrant server
# qdrant_location = ":memory:"             # ephemeral in-process (testing)
# qdrant_location = "/path/to/local"       # persistent local storage
qdrant_collection = "raggity"
# qdrant_api_key = "..."                   # or set QDRANT_API_KEY env var
```

Start a local Qdrant instance with Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Recommended for:** large corpora, multi-user server deployments.

---

## Embedding models

raggity uses [fastembed](https://github.com/qdrant/fastembed) with ONNX Runtime — **CPU by default, no GPU required**.

| Model | Dims | Notes |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | Default — lightweight, portable |
| `nomic-embed-text-v1.5-Q` | 768 | Higher quality — Matryoshka scaling, 8k context |

```toml
[embedding]
model = "BAAI/bge-small-en-v1.5"   # or nomic-embed-text-v1.5-Q
provider = "cpu"                    # cpu / cuda / directml / rocm
```

!!! warning
    Changing `embedding.model` triggers an automatic full index rebuild via the index fingerprint.

## Reranking models

| Model | Size | Notes |
|---|---|---|
| `Xenova/ms-marco-MiniLM-L-6-v2` | ~25 MB | Default — fast and portable |
| `BAAI/bge-reranker-v2-m3` | ~1 GB | Higher quality |

```toml
[retrieval]
rerank_model = "Xenova/ms-marco-MiniLM-L-6-v2"
```
