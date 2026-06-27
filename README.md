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
| `rag ask "..."` | Ask a question; prints the answer with verified source footnotes |
| `rag ask "..." --plain` | Pipe-friendly output — no Rich formatting, no footnotes |
| `rag status` | Show index statistics (chunk count, source count, index path) |
| `rag reindex --force` | Wipe and rebuild the index from scratch |
| `rag eval golden.jsonl` | Run retrieval quality metrics (Hit@k, MRR, Recall@k) against a golden set |

All commands accept `--config PATH` to point at a non-default config file.

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

Endpoints:
- `POST /ingest` — trigger incremental indexing
- `GET /ask?q=...` — ask a question (returns JSON with answer, sources, citations)
- `GET /status` — index statistics

The server respects your `raggity.toml` config, using the same auth, embedding, and retrieval settings as the CLI.

### Streaming

By default, the answer is streamed token-by-token as it is generated — no waiting for the full response. Use `--no-stream` to disable:

```bash
rag ask "..." --no-stream
```

Streaming is supported on both the CLI and the FastAPI server (`/ask` endpoint returns chunked transfer encoding).

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

MIT. See [LICENSE](LICENSE).
