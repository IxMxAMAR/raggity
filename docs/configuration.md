# Configuration

raggity is configured via a `raggity.toml` file in the current directory (or pass `--config PATH` to any command).

Copy the example to get started:

```bash
cp raggity.example.toml raggity.toml
```

---

## Profiles

```toml
profile = "low-ram"
```

A top-level `profile` key selects a named preset. `""` (default) applies no
preset — every field below is governed individually. `"low-ram"` **hard-overrides**
the fields in the table below, even if the same `raggity.toml` also sets those
fields explicitly (the profile always wins). Any other value raises a
validation error naming the valid choices.

| Field | Forced value |
|---|---|
| `index.backend` | `"lancedb"` (embedded — no external vector server to run) |
| `embedding.model` | `"BAAI/bge-small-en-v1.5"` (smallest shipped default) |
| `embedding.cache` | `false` |
| `retrieval.rerank` | `false` (skips loading the cross-encoder model — the biggest RAM save) |
| `retrieval.graph` | `false` |
| `generation.cache` | `false` |
| `server.max_sessions` | `100` |
| `server.max_user_rags` | `4` |

`rag doctor` prints an info line when a profile is active, e.g.:

```
[--] profile: low-ram (rerank/graph/caches off, embedded lancedb)
```

**Measured serve RSS ceiling:** 302 MB peak working set (Windows 11 x64, Python 3.12, raggity v0.11.0, low-ram profile, 5-doc LanceDB index, serving /healthz + /retrieve traffic; embedding model loaded, reranker/graph/caches disabled).

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
rerank_backend = "cross-encoder"
rerank_model = "Xenova/ms-marco-MiniLM-L-6-v2"
colbert_model = "answerdotai/answerai-colbert-small-v1"
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
| `rerank` | `true` | Enable reranking (backend chosen by `rerank_backend`) |
| `rerank_backend` | `"cross-encoder"` | Reranker implementation: `"cross-encoder"` (sigmoid cross-encoder, uses `rerank_model`) or `"colbert"` (late-interaction MaxSim, uses `colbert_model`) |
| `rerank_model` | `"Xenova/ms-marco-MiniLM-L-6-v2"` | ONNX cross-encoder model (used when `rerank_backend = "cross-encoder"`) |
| `colbert_model` | `"answerdotai/answerai-colbert-small-v1"` | fastembed late-interaction model (used when `rerank_backend = "colbert"`) |
| `sufficiency_floor` | `0.5` | Dense-cosine similarity threshold below which raggity abstains ("I don't have enough information") |
| `relevance_floor` | `0.0` | Optional secondary filter on the rerank score (0.0 = off); does not trigger abstention. Not comparable across `rerank_backend` values — see below |
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
| `corrective` | `false` | Enable CRAG-style corrective retrieval: a retrieval evaluator (+1 LLM call/question) plus one query-rewrite-and-merge round (+1 more when triggered) — see [Corrective retrieval](retrieval.md#corrective-retrieval-crag-style) |

### Heavy reranker

```toml
[retrieval]
rerank_model = "BAAI/bge-reranker-v2-m3"   # ~1 GB, higher quality
```

### Late-interaction reranking

`rerank_backend = "colbert"` swaps the cross-encoder for a ColBERT-style late-interaction
reranker (fastembed's `LateInteractionTextEmbedding`). Instead of scoring the whole
query against the whole chunk in one pass, it embeds the query and each candidate as
*per-token* vectors and scores each chunk with MaxSim (for every query token, the best
matching chunk token; scores summed and normalized by query-token count). This gives
finer-grained, query-token-level matching than a single cross-encoder pass — often
better at catching a chunk that only satisfies part of a multi-clause query — at the
cost of an extra ~0.13 GB model download on first use (the default
`answerdotai/answerai-colbert-small-v1`) and somewhat more compute per rerank call.
It is storage-free: no index changes, rerank-stage only.

```toml
[retrieval]
rerank_backend = "colbert"
```

!!! warning
    ColBERT MaxSim scores are normalized to roughly `[0, 1]` but are **not** on the
    same scale as the cross-encoder's sigmoid scores. A `relevance_floor` tuned for
    one `rerank_backend` does not transfer to the other — re-tune it if you switch.

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

# Externally-managed OpenAI-compatible backend
# backend = "external"
# model = "some-model"
# base_url = "http://127.0.0.1:9999/v1"

# Opt-in personalization (default off). See "Personalization" below.
# persona = "The user is Dr. Vane, a cardiologist. Prefer clinical phrasing."
# personal_kb = false

# Rolling conversation-summary memory for rag chat / /chat sessions. See
# "Conversation memory" below. 0 disables summarization.
# memory_max_turns = 20
```

| Key | Default | Description |
|---|---|---|
| `backend` | `"claude"` | LLM backend: `"claude"`, `"openai"`, `"ollama"`, or `"external"` |
| `model` | `"claude-opus-4-8"` | Model name (backend-specific) |
| `auth` | `"auto"` | Claude auth mode: `"auto"`, `"subscription"`, or `"api_key"` |
| `cache` | `false` | Semantic answer cache (keyed on question + chunks + model + effective system prompt) |
| `temperature` | `null` | Generation temperature passed to the model (null = use model default) |
| `base_url` | varies | OpenAI-compatible base URL |
| `api_key_env` | `"OPENAI_API_KEY"` | Env var name holding the API key (OpenAI/Ollama backends) |
| `auto_start` | `true` | Auto-start a local backend (e.g. ollama) on first use when a runtime binary is found and the server is not already running |
| `persona` | `""` | Free-form user context appended to the system prompt (grounding rules still bind). Empty = system prompt unchanged |
| `personal_kb` | `false` | Treat the knowledge base as the current user's own (first-person docs/questions refer to them) |
| `memory_max_turns` | `20` | Chat turn count above which the oldest turns are compressed into a rolling summary (see "Conversation memory" below). `0` disables summarization — chat falls back to the fixed 6-turn prompt window only |

### Personalization

`persona` and `personal_kb` are **opt-in** and off by default; with both unset the
system prompt is byte-identical to the stock one. When set, their content is
appended to the system prompt as a clearly-delimited **User context:** block,
followed by a reminder that the citation + abstention rules still apply — the
model must still answer only from the retrieved context and cite every claim.
Toggling either value changes the answer-cache key, so cached answers are
invalidated automatically.

### Conversation memory

`rag chat` and server chat sessions always prompt the model with only the
most-recent 6 turns (`Conversation.recent(6)`) — that window never grows.
Beyond that, once a conversation accumulates more than `memory_max_turns`
total turns, the oldest turns (everything beyond the most-recent
`memory_max_turns // 2`) are compressed into a rolling summary via **one**
LLM call and dropped from the turn list; the summary is carried forward and
merged with on each subsequent overflow. The summary is injected as a
synthetic leading line ("Earlier conversation summary: ...") in the prompt's
conversation-so-far block, so long chats keep earlier context without an
ever-growing prompt. If the summarization call fails, the oldest turns are
still dropped (plain truncation) so the conversation stays bounded — the
chat itself never raises. Set `memory_max_turns = 0` to disable
summarization entirely (pure fixed-window behavior, the old default).

### Local providers: discovery & auto-start

`rag model --list` probes known local runtimes (ollama, lmstudio, llamacpp,
vllm, jan, koboldcpp) and prints which are running, installed, and what models
they expose — copy-pasteable into `rag model <model> -p <provider>`. Local
OpenAI-compatible runtimes map to `backend = "openai"` plus the runtime's
default `base_url` (no API key required for a loopback server); `ollama` keeps
`backend = "ollama"`. With `auto_start = true`, an `ollama` backend is started
on first request if the `ollama` binary is found and the server is not already
up. `rag doctor` reports the full discovery table and, for ollama, will start
the server and check that the configured model is pulled.

### Externally-managed backend (`backend = "external"`)

`backend = "external"` targets an OpenAI-compatible server whose lifecycle is
owned by another tool — for example [Rigma](https://github.com/IxMxAMAR/rigma)
or a server you start and manage yourself. raggity **never** auto-starts it,
even if a runtime binary is discoverable on the machine.

```toml
[generation]
backend = "external"
model = "some-model"
base_url = "http://127.0.0.1:9999/v1"   # required — no default
```

`base_url` is **required**; raggity raises at startup if it's missing.
Readiness is checked lazily on first request (and by `rag doctor`) via `GET
<root>/health`, falling back to `GET <root>/v1/models` if that fails. If both
probes fail, raggity raises a clear error naming the exact `base_url` and
stating that `backend=external` never launches servers itself.

Switch to it from the CLI:

```bash
rag model some-model -p external --base-url http://127.0.0.1:9999
```

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
