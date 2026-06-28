# raggity

Local-first, top-tier RAG over your notes, docs, and PDFs — answered by Claude.

Hybrid retrieval (dense + BM25 + RRF), cross-encoder reranking, dedup, verified inline citations, and selective abstention: raggity only answers when it has evidence.

---

## Install

```bash
pip install raggity
```

Python 3.11+ required. See [Platform support](#platform-support) for OS/architecture notes.

---

## Auth

raggity answers questions via the **Claude Agent SDK**, using your Claude subscription or API key.

### Subscription (default — recommended)

Log in once with the Claude CLI:

```bash
claude login
```

Leave `generation.auth = "auto"` in your config (the default). raggity will use your subscription session when no `ANTHROPIC_API_KEY` is set — no API key needed.

### API key

If you prefer to use an API key instead:

1. Set the environment variable:

   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   ```

2. Set `auth = "api_key"` in `raggity.toml` under `[generation]`.

- `auth = "subscription"` — uses your `claude login` session. **`ANTHROPIC_API_KEY` is intentionally ignored**, even if set; the SDK cannot fall back to a per-token key.
- `auth = "api_key"` — requires `ANTHROPIC_API_KEY` to be set; raises an error at startup if missing.
- `auth = "auto"` (default) — uses a key if `ANTHROPIC_API_KEY` is present, otherwise falls back to the subscription session.

---

## Generation backends

raggity supports three generation backends, controlled by `generation.backend` in `raggity.toml`.

### Claude (default)

Uses the Claude Agent SDK. No extra dependencies. Requires a Claude subscription or API key (see [Auth](#auth) above).

```toml
[generation]
backend = "claude"
model = "claude-opus-4-8"
auth = "auto"
```

### OpenAI-compatible

Any OpenAI-compatible API endpoint (OpenAI, Azure OpenAI, Together, Groq, etc.). Requires the `openai` extra:

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

### Ollama (offline)

Runs against a local [Ollama](https://ollama.com) server — no API key required. Requires the `openai` extra (reuses the OpenAI client):

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

The `auth` field is ignored for `openai` and `ollama` backends.

---

## Quickstart

```bash
# 1. Copy the example config and edit it
cp raggity.example.toml raggity.toml
```

Open `raggity.toml` and point `sources.include` at your notes:

```toml
[sources]
include = ["~/notes/**/*.md", "~/docs/**/*.pdf"]
```

```bash
# 2. Index your sources (incremental — safe to re-run)
rag ingest

# 3. Ask a question
rag ask "How do I set up a new dev environment?"
```

Both `rag` and `raggity` are registered as console scripts — they are identical aliases. Use whichever you prefer.

---

## Commands

| Command | Description |
|---|---|
| `rag ingest` | Incrementally index configured sources (hash-based, only processes changes) |
| `rag ingest-url <url>` | Fetch a web URL (and optionally crawl same-domain links) and add to the index |
| `rag ingest-repo <url>` | Shallow-clone a git repository and index all text files in it |
| `rag ingest-obsidian <vault>` | Read all Markdown notes from an Obsidian vault and add to the index |
| `rag ask "..."` | Ask a question; prints the answer with verified source footnotes |
| `rag ask "..." --plain` | Pipe-friendly output — no Rich formatting, no footnotes |
| `rag ask "..." --hyde` | HyDE query transform — generate a hypothetical passage to improve dense recall |
| `rag ask "..." --step-back` | Step-back query transform — generate a broader question for higher-level context |
| `rag ask "..." --decompose` | Decompose into sub-questions, retrieve for each, merge and answer |
| `rag ask "..." --no-cache` | Bypass the answer cache even when `generation.cache = true` |
| `rag chat` | Start an interactive multi-turn chat REPL in the terminal |
| `rag serve` | Start the local HTTP API server |
| `rag serve --open` | Start the server and open the web chat UI in your default browser |
| `rag status` | Show index statistics (chunk count, source count, index path) |
| `rag reindex --force` | Wipe and rebuild the index from scratch |
| `rag eval golden.jsonl` | Run retrieval quality metrics (Hit@k, MRR, Recall@k) against a golden set |
| `rag eval golden.jsonl --llm-judge` | LLM-judge eval: faithfulness + answer relevance (2 model calls per question; self-assessed) |
| `rag watch` | Watch source folders and re-index automatically on file changes (Ctrl-C to stop) |

All commands accept `--config PATH` to point at a non-default config file.

---

## Supported file types

raggity reads the following file types out of the box:

| Extension | Notes |
|---|---|
| `.md` | Markdown |
| `.txt` | Plain text |
| `.pdf` | Embedded text extraction via pypdf; falls back to OCR when text is absent |
| `.docx` | Requires `raggity[docs]` |
| `.html` | Requires `raggity[docs]` |
| `.csv` | Parsed as key: value rows |
| `.pptx` | Requires `raggity[docs]` |
| `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp` | OCR via RapidOCR — requires `raggity[ocr]` |

### Document extras

Install optional readers with the `docs` extra:

```bash
pip install raggity[docs]
```

This adds support for `.docx`, `.html`, and `.pptx` files.

### OCR extra

For scanned PDFs and image files, install the `ocr` extra:

```bash
pip install raggity[ocr]
```

This adds RapidOCR + pypdfium2.  raggity will automatically OCR a PDF page when embedded text is absent.

---

## Connectors

Connectors let you ingest content from external sources beyond your local file system.

### Web (`rag ingest-url`)

Requires the `web` extra:

```bash
pip install raggity[web]
```

```bash
# Fetch a single page
rag ingest-url https://docs.example.com/overview

# BFS-crawl same-domain links up to 2 hops deep
rag ingest-url https://docs.example.com --depth 2
```

You can also configure URLs for automatic ingestion on every `rag ingest` run:

```toml
[sources]
urls = ["https://docs.example.com/overview", "https://example.com/changelog"]
```

### GitHub / Git repo (`rag ingest-repo`)

No extra install needed — uses stdlib subprocess + your local `git`.

```bash
# Index the default branch of a GitHub repo
rag ingest-repo https://github.com/owner/repo

# Pin to a specific branch or tag
rag ingest-repo https://github.com/owner/repo --ref main
```

All text files with supported extensions are read and indexed.  The index
`path` for each document is ``<repo_url>#<relpath>`` so you can trace sources
back to the repository.

### Obsidian vault (`rag ingest-obsidian`)

No extra install needed.

```bash
rag ingest-obsidian ~/Documents/MyVault
```

raggity walks all `.md` files in the vault recursively and normalises
``[[wikilink]]`` / ``[[link|alias]]`` syntax to plain text before indexing,
so bracket noise does not pollute your chunks.

---

## CPU / AMD / NVIDIA portability

Embeddings and reranking run on **CPU by default** via ONNX Runtime — no GPU required, no CUDA setup. This works on all supported platforms out of the box.

If you want GPU acceleration, set `provider` in `[embedding]`:

```toml
[embedding]
provider = "cpu"        # default — works everywhere
# provider = "cuda"     # NVIDIA (CUDA 11/12)
# provider = "directml" # Windows — AMD, Intel, NVIDIA via DirectML
# provider = "rocm"     # Linux — AMD ROCm
```

The provider switch affects only the embedding and reranking inference — the rest of the pipeline (retrieval, generation) is unaffected. Claude generation always runs in the cloud via the SDK.

---

## Retrieval pipeline

```
Sources → Chunker → Embedder → LanceDB
                                  |
          Query ──────────────────┤
                    dense search  ├── RRF fusion (k=60)
                    BM25/FTS      ┘       |
                                    Cross-encoder rerank
                                          |
                                    Dedup (cosine >= 0.92)
                                          |
                                    Relevance floor filter
                                          |
                                    Lost-in-the-middle reorder
                                          |
                                    Claude Agent SDK → Answer
                                    (with verified citations)
```

- **Hybrid retrieval**: dense vector search + BM25 full-text search, fused with Reciprocal Rank Fusion (RRF, k=60).
- **Cross-encoder reranking**: enabled by default (`rerank = true`). Uses a local ONNX cross-encoder to re-score candidate chunks against the query before selection.
- **Deduplication**: chunks with cosine similarity >= `dedup_cosine` (default 0.92) are collapsed.
- **Relevance floor**: when `rerank = true`, chunks whose sigmoid-normalised cross-encoder score falls below `relevance_floor` are dropped before generation. The floor is **not applied** when `rerank = false` — abstention then triggers only when retrieval returns no candidates at all.
- **Lost-in-the-middle reorder**: top chunks are placed at the start and end of the context window where LLMs attend best.
- **Selective abstention**: if no chunk clears the relevance floor, raggity returns a canned "I don't have enough information" message without calling the API.
- **Verified citations**: inline citation markers in the answer are cross-checked against the retrieved chunks; only markers that match a real retrieved source are preserved.

---

## Tuning

`relevance_floor` is the main knob. When `rerank = true` (default), cross-encoder logits are sigmoid-normalised to (0, 1) before comparison; the default 0.3 is a good starting point. When `rerank = false`, the floor is not applied at all — candidates pass straight to dedup and top-k selection regardless of their score.

Other useful knobs in `[retrieval]`:

| Key | Default | Notes |
|---|---|---|
| `candidates` | 30 | Chunks fetched from each retriever before fusion |
| `top_k` | 5 | Chunks passed to the LLM after all filtering |
| `dedup_cosine` | 0.92 | Cosine threshold for dedup collapse |
| `rrf_k` | 60 | RRF constant (higher = flatter fusion curve) |

---

## Phase 2.5 features

### HyDE (Hypothetical Document Embeddings)

The `--hyde` flag generates a hypothetical answer passage using Claude, then uses that passage (alongside the original question) as an additional query vector. This improves dense retrieval recall for questions where the answer vocabulary differs from the question vocabulary.

```bash
rag ask "What are the main tradeoffs of eventual consistency?" --hyde
```

Enable permanently in config:
```toml
[retrieval]
hyde = true
```

### Step-back prompting

The `--step-back` flag generates a broader, more abstract "step-back" question using Claude, retrieves for it alongside the original question, and merges the results. Useful for grounding specific questions in general principles.

```bash
rag ask "How do I configure the database connection pool?" --step-back
```

Enable permanently:
```toml
[retrieval]
step_back = true
```

Both `--hyde` and `--step-back` add one model call each. They compose freely with `--expand` — all generated queries are fused via RRF.

### Decompose

The `--decompose` flag breaks a complex question into sub-questions using Claude, retrieves chunks for each sub-question independently, merges the candidates by chunk ID (dedup), and answers over the combined context. This is useful for multi-faceted questions that benefit from different retrieval angles.

```bash
rag ask "How do backups, retention policies, and restore procedures interact?" --decompose
```

`--decompose` overrides `--hyde`, `--step-back`, and `--expand` when combined.

### Semantic answer cache

When `generation.cache = true`, raggity stores answers in `<index.path>/answer_cache.json`, keyed on a SHA-256 hash of the question + retrieved chunk IDs (order-independent) + model name. On a cache hit, the stored answer is returned immediately — no model call.

```toml
[generation]
cache = true
```

The cache is **off by default** — existing workflows are unaffected. To bypass the cache for a single query:

```bash
rag ask "..." --no-cache
```

Notes:
- Cache hits are exact: same question + same retrieved chunks + same model.
- The streaming path (`rag ask` without `--no-stream`) always calls the model; only the buffered path (`--no-stream` or `--plain`) reads/writes the cache.
- The cache file is a plain JSON file and can be deleted to clear it.

### LLM-judge eval

```bash
rag eval golden.jsonl --llm-judge
```

Reports `Faithfulness` (does the answer stay grounded in the retrieved chunks?) and `AnswerRelevance` (does the answer address the question?), each in [0, 1]. Each question costs two model calls.

**Caveat:** self-assessed — the same model family that generates answers also grades them. Treat these scores as a sanity check, not ground truth.

---

## Scalability

### Watch daemon

Install the watch extra:

```bash
pip install raggity[watch]
```

Start the watch daemon:

```bash
rag watch
```

raggity monitors all paths in `sources.include` recursively. When files change, it triggers a debounced re-index (default 2 s of quiet before the ingest runs, coalescing rapid filesystem events into a single call).

Use the `--debounce` CLI flag to customize the delay:

```bash
rag watch --debounce 5.0   # wait 5 s of quiet before re-indexing
```

The daemon runs until you press Ctrl-C.

### Qdrant backend

For large-scale or multi-user deployments, raggity supports Qdrant as an alternative vector store:

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

LanceDB (the default) requires no extra install and is the recommended choice for single-user local deployments.

### Batch and parallel embedding

raggity automatically batches embedding calls (default `batch_size = 256`). Parallel workers (`parallel = 0` = auto) are used when supported by the model. Configure in `raggity.toml`:

```toml
[embedding]
batch_size = 256   # increase for faster ingest on large corpora
parallel = 0       # number of parallel embedding workers (0 = auto)
```

### Embedding cache

To avoid re-embedding unchanged chunks across ingest runs, enable the embedding cache:

```toml
[embedding]
cache = true       # cache embeddings as JSON under the index directory
```

Cached embeddings are stored as JSON at `<index.path>/embed_cache.json` and are looked up by content hash before calling the embedding model — useful when you frequently run `rag ingest` on large corpora with small diffs.

### ANN auto-index

raggity automatically builds an Approximate Nearest Neighbor (ANN) index on the vector store once the collection grows past a threshold (default 50 000 chunks). This keeps search latency flat as your knowledge base scales. Configure the threshold:

```toml
[index]
ann_threshold = 50000   # build ANN index once chunk count exceeds this
```

---

## Phase 2 features

### Parent-document retrieval

By default, raggity chunks documents into fixed-size pieces (256 tokens) for indexing. When `parent_document = true`, each chunk retains a reference to its parent document (up to 1024 tokens), and retrieval expands matched chunks to include their parents when passing context to Claude.

```toml
[retrieval]
parent_document = true
```

Parent-document mode automatically rebuilds the index via the index fingerprint when enabled or disabled — no manual steps needed.

### Query expansion

The `--expand` flag generates multiple query variations using Claude, then retrieves for each and reranks the combined results using Reciprocal Rank Fusion. This improves coverage for complex questions but increases API calls.

```bash
rag ask "How do I set up a new dev environment?" --expand
```

Expansion uses the `generation.model` to generate variations; the number of variations is configurable via `retrieval.expand_n` (default 3).

### FastAPI server

Install the server extras:

```bash
pip install raggity[server]
```

Start the server:

```bash
rag serve
```

Open the web chat UI automatically:

```bash
rag serve --open
```

`--open` calls `webbrowser.open(http://host:port)` just before the server starts — best-effort, guarded, no error if a browser cannot be launched.

#### Web UI

`GET /` serves a self-contained single-page chat UI (vanilla JS, no build step, no CDN required). It streams answers from `/ask/stream` via `EventSource`, keeps a `session_id` in memory across the tab's lifetime, and renders a "Sources" list from the terminal SSE `done` event.

#### REST endpoints

- `GET /` — web chat UI (HTML)
- `POST /ingest` — trigger incremental indexing
- `POST /ask` — ask a question (optionally with a `session_id` to enable multi-turn history); returns JSON with `answer`, `abstained`, `citations`, and optionally `session_id`
- `POST /chat` — stateful chat endpoint; always creates or reuses a session; returns `session_id`
- `GET /ask/stream?question=...&session_id=...` — SSE streaming answer; yields `data:` delta chunks, then a terminal `event: done` with a JSON payload containing `citations` (and `session_id` when provided)
- `DELETE /session/{id}` — discard a conversation session
- `GET /status` — index statistics

#### Server sessions

Pass `session_id` on any `/ask` or `/ask/stream` request to thread conversation history across turns. The session lives in server memory for the lifetime of the process. Use `DELETE /session/{id}` to clear it.

Example:
```bash
# Stateless single-turn ask
curl -X POST http://localhost:8000/ask \
  -H "content-type: application/json" \
  -d '{"question": "How do I set up a new dev environment?"}'

# Multi-turn session
SID=$(python -c "import uuid; print(uuid.uuid4().hex)")
curl -X POST http://localhost:8000/chat \
  -H "content-type: application/json" \
  -d "{\"question\": \"What are backups?\", \"session_id\": \"$SID\"}"

# SSE streaming
curl -N "http://localhost:8000/ask/stream?question=What+are+backups%3F&session_id=$SID"
```

The server respects your `raggity.toml` config, using the same auth, embedding, and retrieval settings as the CLI.

### Terminal chat REPL

```bash
rag chat
```

Starts an interactive multi-turn conversation in your terminal. Type a question and press Enter; raggity streams the answer token-by-token with verified source footnotes. Type `exit` or press Ctrl-D to quit.

### Streaming

The CLI `rag ask` command streams the answer token-by-token as it is generated by default — no waiting for the full response. Use `--no-stream` to disable:

```bash
rag ask "..." --no-stream
```

The FastAPI server `/ask` endpoint returns the full answer as a single JSON response (not streamed); only the CLI streams tokens.

### Heavy reranker and nomic embed options

The default models (`Xenova/ms-marco-MiniLM-L-6-v2` for reranking, `BAAI/bge-small-en-v1.5` for embedding) are lightweight and portable. For higher quality, you can switch to heavier alternatives:

**Heavy reranker** (BAAI/bge-reranker-v2-m3, ~1GB):
```toml
[retrieval]
rerank_model = "BAAI/bge-reranker-v2-m3"
```

**Larger embedding model** (nomic-embed-text-v1.5-Q, 768-dim with Matryoshka scaling and 8k context):
```toml
[embedding]
model = "nomic-embed-text-v1.5-Q"
```

Changing `embedding.model` or `parent_document` triggers an automatic full rebuild via the index fingerprint. Heavy models download a large file on first use; this happens transparently during the first `rag ingest` or `rag ask` command after the config change.

---

## Platform support

| Platform | Status |
|---|---|
| Linux x86_64 | Supported |
| macOS Apple Silicon (ARM) | Supported |
| Windows x64 | Supported |
| Intel macOS x86_64 | Partial — onnxruntime wheels may be missing for some Python versions; LanceDB may have build issues |
| Windows on ARM | Partial — onnxruntime and LanceDB binary wheels are not consistently available |

If you hit a missing wheel on an unsupported platform, open an issue or try building from source. The CPU-only path (no GPU provider) is the most portable.

---

## Roadmap

**Phase 2 (deferred):** web/RSS sources, streaming `rag ask`, watch-mode auto-reindex, configurable chunking strategies, conversation history.

**Phase 3 (scalability):** distributed index backends (remote LanceDB / Pinecone / Weaviate), multi-user deployments, async ingestion pipeline, document-level access control.

---

## License

**GNU AGPL-3.0-or-later.** See [LICENSE](LICENSE). This is a strong copyleft license:
if you modify raggity and distribute it — or run a modified version as a network/hosted
service — you must release your source under the AGPL as well. Using raggity as-is to query
your own documents has no such obligation.
