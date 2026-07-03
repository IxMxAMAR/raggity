# Configuration

raggity is configured via a `raggity.toml` file in the current directory (or pass `--config PATH` to any command).

Copy the example to get started:

```bash
cp raggity.example.toml raggity.toml
```

---

## `[sources]`

```toml
[sources]
include = ["~/notes/**/*.md", "~/docs/**/*.pdf"]
exclude = ["**/drafts/**", "**/*.tmp.md"]
urls = ["https://docs.example.com/overview"]
```

| Key | Default | Description |
|---|---|---|
| `include` | `[]` | Glob patterns for local files to index |
| `exclude` | `[]` | Glob patterns (fnmatch on the posix path) to skip. Applied on top of built-in junk-dir pruning |
| `urls` | `[]` | URLs to fetch and index on every `rag ingest` run |

> Built-in junk directories are **always** pruned when they appear *below* an
> `include` pattern's static (pre-glob) prefix: `AppData`, `node_modules`,
> `.git`, `__pycache__`, `site-packages`, `.venv`, `venv`, `dist-packages`,
> `.raggity`, `.npm`, `.nuget`, `.gradle`, `.cargo`, `.conda`. This stops a
> broad pattern like `**/*.txt` run from your home directory from sweeping
> caches and dependency trees. A pattern pointed *inside* such a dir still works.

---

## `[index]`

```toml
[index]
path = ".raggity/index"       # relative to the current working directory
backend = "lancedb"           # or "qdrant"
ann_threshold = 50000         # build ANN index once chunk count exceeds this

# Qdrant-specific (only when backend = "qdrant")
# qdrant_location = ":memory:"             # default — ephemeral in-memory
# qdrant_location = "http://localhost:6333" # served Qdrant instance
qdrant_collection = "raggity"
# qdrant_api_key = "..."      # or set QDRANT_API_KEY env var
```

| Key | Default | Description |
|---|---|---|
| `path` | `".raggity/index"` | Directory for LanceDB data and caches (relative to cwd) |
| `backend` | `"lancedb"` | Vector store backend: `"lancedb"` or `"qdrant"` |
| `ann_threshold` | `50000` | Chunk count above which ANN index is built automatically |
| `qdrant_location` | `":memory:"` | Qdrant location: `":memory:"` (default, ephemeral in-process), `"http://localhost:6333"` (served instance), or a local path |
| `qdrant_collection` | `"raggity"` | Qdrant collection name |
| `qdrant_api_key` | `""` | Qdrant API key (or set `QDRANT_API_KEY`) |

---

## `[embedding]`

```toml
[embedding]
model = "BAAI/bge-small-en-v1.5"   # default lightweight model
provider = "cpu"                    # cpu / cuda / directml / rocm
batch_size = 256
# parallel = 4                      # omit for in-process (default); N = worker pool
cache = false                       # cache embeddings as JSON
```

| Key | Default | Description |
|---|---|---|
| `model` | `"BAAI/bge-small-en-v1.5"` | fastembed model name |
| `provider` | `"cpu"` | ONNX Runtime execution provider |
| `batch_size` | `256` | Embedding batch size |
| `parallel` | _(unset)_ | Embedding worker pool size. Default (unset/`None`) = in-process single model, the stable path. A positive `N` spawns `N` multiprocessing workers (each loads its own ONNX model); avoid on memory-constrained Windows |
| `cache` | `false` | Cache embeddings by content hash to avoid re-embedding unchanged chunks |

### GPU acceleration

```toml
[embedding]
provider = "cpu"        # default — works everywhere
# provider = "cuda"     # NVIDIA (CUDA 11/12)
# provider = "directml" # Windows — AMD, Intel, NVIDIA via DirectML
# provider = "rocm"     # Linux — AMD ROCm
```

### Larger embedding model

For higher embedding quality (768-dim, 8k context, Matryoshka scaling):

```toml
[embedding]
model = "nomic-embed-text-v1.5-Q"
```

!!! warning
    Changing `embedding.model` triggers an automatic full index rebuild.

---

## `[retrieval]`

```toml
[retrieval]
candidates = 30
top_k = 5
rerank = true
rerank_model = "Xenova/ms-marco-MiniLM-L-6-v2"
sufficiency_floor = 0.5   # dense-cosine threshold — governs abstention
relevance_floor = 0.0     # optional rerank-score filter (0.0 = off)
hybrid = true
dedup_cosine = 0.92
rrf_k = 60
parent_document = false
hyde = false
step_back = false
expand_n = 3
graph = false
graph_hops = 1
```

| Key | Default | Description |
|---|---|---|
| `candidates` | `30` | Chunks fetched from each retriever before fusion |
| `top_k` | `5` | Chunks passed to the LLM after all filtering |
| `rerank` | `true` | Enable cross-encoder reranking |
| `rerank_model` | `"Xenova/ms-marco-MiniLM-L-6-v2"` | ONNX cross-encoder model |
| `sufficiency_floor` | `0.5` | Dense-cosine similarity threshold below which raggity abstains ("I don't have enough information") |
| `relevance_floor` | `0.0` | Optional secondary filter on the sigmoid-normalised cross-encoder rerank score (0.0 = off); does not trigger abstention |
| `hybrid` | `true` | Enable hybrid (dense + BM25) retrieval |
| `dedup_cosine` | `0.92` | Cosine similarity threshold for chunk deduplication |
| `rrf_k` | `60` | RRF fusion constant (higher = flatter curve) |
| `parent_document` | `false` | Expand matched chunks to parent documents |
| `hyde` | `false` | Enable HyDE query transform permanently |
| `step_back` | `false` | Enable step-back query transform permanently |
| `expand_n` | `3` | Number of query variations for `--expand` |
| `graph` | `false` | Enable GraphRAG knowledge-graph augmentation |
| `graph_hops` | `1` | BFS hops from matched entities in the graph |
| `graph_concurrency` | `8` | Concurrent LLM extraction calls during `rag graph-build` (lower it for strict-rate-limit backends) |

### Heavy reranker

```toml
[retrieval]
rerank_model = "BAAI/bge-reranker-v2-m3"   # ~1 GB, higher quality
```

---

## `[generation]`

```toml
[generation]
backend = "claude"
model = "claude-opus-4-8"
auth = "auto"
cache = false

# OpenAI-compatible backend
# backend = "openai"
# model = "gpt-4o-mini"
# base_url = "https://api.openai.com/v1"
# api_key_env = "OPENAI_API_KEY"

# Ollama backend
# backend = "ollama"
# model = "llama3.1"
# base_url = "http://localhost:11434/v1"
```

| Key | Default | Description |
|---|---|---|
| `backend` | `"claude"` | LLM backend: `"claude"`, `"openai"`, or `"ollama"` |
| `model` | `"claude-opus-4-8"` | Model name (backend-specific) |
| `auth` | `"auto"` | Claude auth mode: `"auto"`, `"subscription"`, or `"api_key"` |
| `cache` | `false` | Semantic answer cache (keyed on question + chunks + model) |
| `temperature` | `null` | Generation temperature passed to the model (null = use model default) |
| `base_url` | varies | OpenAI-compatible base URL |
| `api_key_env` | `"OPENAI_API_KEY"` | Env var name holding the API key (OpenAI/Ollama backends) |

---

## `[server]`

```toml
[server]
host = "0.0.0.0"
port = 8000
```

| Key | Default | Description |
|---|---|---|
| `host` | `"0.0.0.0"` | Server bind host |
| `port` | `8000` | Server listen port |

---

## `[observability]`

```toml
[observability]
otel_endpoint = ""    # OTLP gRPC endpoint, e.g. "http://localhost:4317"
```

| Key | Default | Description |
|---|---|---|
| `otel_endpoint` | `""` | OpenTelemetry collector endpoint. Empty = disabled |

Requires the `otel` extra: `pip install raggity[otel]`.
