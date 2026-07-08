# Server & API

raggity includes a FastAPI HTTP server with a web chat UI, SSE streaming, and multi-turn sessions.

---

## Install

```bash
pip install raggity[server]
```

---

## Starting the server

```bash
rag serve
```

Open the web chat UI automatically in your default browser:

```bash
rag serve --open
```

The server reads your `raggity.toml` and uses the same auth, embedding, and retrieval settings as the CLI.

---

## Web chat UI

`GET /` serves a **self-contained single-page chat UI** — vanilla JS, no build step, no CDN required. It:

- Streams answers from `/ask/stream` via `EventSource`
- Keeps a `session_id` in memory across the tab's lifetime
- Renders a "Sources" list from the terminal SSE `done` event

---

## REST endpoints

### `GET /`

Returns the web chat UI (HTML).

### `POST /ingest`

Trigger incremental indexing.

```bash
curl -X POST http://localhost:8000/ingest
```

### `POST /ingest/content`

Ingest caller-supplied documents directly (no server-side file access needed). Requires auth when enabled, and ingests into the **caller's own namespace** when `server.per_user = true`. Body: `{"documents": [{"path": "...", "text": "...", "title": "..."}]}`. Returns `{"ingested": <n>}`.

```bash
curl -X POST http://localhost:8000/ingest/content \
  -H "content-type: application/json" \
  -H "X-API-Key: $KEY" \
  -d '{"documents": [{"path": "note1.md", "text": "Backups run nightly at 2am."}]}'
```

### `POST /ask`

Ask a question. Returns JSON with `answer`, `abstained`, `citations`, and optionally `session_id`.

```bash
# Stateless single-turn
curl -X POST http://localhost:8000/ask \
  -H "content-type: application/json" \
  -d '{"question": "How do I set up a new dev environment?"}'
```

With a session:

```bash
curl -X POST http://localhost:8000/ask \
  -H "content-type: application/json" \
  -d '{"question": "What are backups?", "session_id": "abc123"}'
```

### `POST /chat`

Stateful chat endpoint. Always creates or reuses a session. Returns `session_id`.

```bash
SID=$(python -c "import uuid; print(uuid.uuid4().hex)")
curl -X POST http://localhost:8000/chat \
  -H "content-type: application/json" \
  -d "{\"question\": \"What are backups?\", \"session_id\": \"$SID\"}"
```

### `GET /ask/stream`

SSE streaming answer. Yields `data:` delta chunks as tokens arrive, then a terminal `event: done` with a JSON payload containing `citations` (and `session_id` when provided).

```bash
# Stream a response
curl -N "http://localhost:8000/ask/stream?question=What+are+backups%3F"

# Stream with session
curl -N "http://localhost:8000/ask/stream?question=What+are+backups%3F&session_id=$SID"
```

SSE event format:

```
:ok

data: Hello

data:  world

event: done
data: {"citations": ["source.md"], "session_id": "abc123"}
```

Both the stateless and the session (`session_id`) paths stream true incremental token deltas. The stream opens with a `:ok` comment and is sent with `Cache-Control: no-cache` and `X-Accel-Buffering: no`, so it passes through nginx-style reverse proxies without being buffered. Newlines inside a delta are framed as consecutive `data:` lines per the SSE spec.

### `DELETE /session/{id}`

Discard a conversation session from server memory.

```bash
curl -X DELETE http://localhost:8000/session/abc123
```

### `GET /status`

Index statistics (chunk count, source count, index path).

```bash
curl http://localhost:8000/status
```

### `GET /healthz`

**Unauthenticated** liveness probe — no `X-API-Key`/`Authorization` header required, even when `server.auth = "api_key"`. It is deliberately cheap: it only touches the vector store (`store.count()`) and never loads the embedding model, reranker, or LLM provider, so it responds in about 2 ms and keeps working even when the generation backend is completely unreachable.

```bash
curl http://localhost:8000/healthz
```

```json
{
  "status": "ok",
  "version": "0.11.0",
  "index_backend": "lancedb",
  "documents": 1234
}
```

`documents` is the **chunk count**, not the source-document count (the work-order contract for this endpoint). Under `server.per_user = true` it always reports the shared **base** index — it is a liveness signal, not a per-tenant statistic; use `GET /status` (authenticated) for per-tenant numbers.

### `POST /retrieve`

Retrieval only — no LLM call is made and no query transforms run, so it keeps working even when the generation backend is unreachable. Returns the top-`k` chunks plus a greedily-packed context string, so external orchestrators (agents, other tools) can do their own relevance judgment and generation.

Request:

```json
{"query": "How do backups work?", "k": 8, "max_context_tokens": 2000}
```

| Field | Default | Description |
|---|---|---|
| `query` | required | Search query |
| `k` | `8` (`>= 1`) | Number of chunks to retrieve |
| `max_context_tokens` | `null` (`>= 1` when set) | Optional budget for `packed_context`; unset packs all retrieved chunks |

Response:

```json
{
  "chunks": [
    {
      "text": "Backups run nightly at 2am...",
      "score": 0.83,
      "source": "notes/ops.md",
      "metadata": {
        "title": "Ops notes",
        "heading_path": "Backups",
        "chunk_id": "abc123",
        "ordinal": 0
      }
    }
  ],
  "packed_context": "[source: notes/ops.md]\nBackups run nightly at 2am...",
  "token_count": 42,
  "tokenizer": "chars/4-approx"
}
```

`packed_context` joins `[source: <path>]\n<text>` blocks with blank lines, greedily packed to `max_context_tokens` (when set) using the `chars/4` approximate tokenizer reported in `tokenizer`.

By design this endpoint **bypasses the sufficiency-floor abstention** used by `/ask` and `/chat` — callers get the raw top-k with scores and judge relevance themselves rather than raggity silently withholding low-confidence results.

`/retrieve` is authenticated the same way as the other data routes (respects `server.auth`) and is per-tenant aware under `server.per_user = true`.

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "content-type: application/json" \
  -H "X-API-Key: $KEY" \
  -d '{"query": "How do backups work?", "k": 8, "max_context_tokens": 2000}'
```

---

## Sessions

Pass `session_id` on any `/ask` or `/ask/stream` request to thread conversation history across turns. The session lives in **server memory** for the lifetime of the process.

```bash
# Create a session ID
SID=$(python -c "import uuid; print(uuid.uuid4().hex)")

# Turn 1
curl -X POST http://localhost:8000/chat \
  -H "content-type: application/json" \
  -d "{\"question\": \"What is raggity?\", \"session_id\": \"$SID\"}"

# Turn 2 — context from turn 1 is included
curl -X POST http://localhost:8000/chat \
  -H "content-type: application/json" \
  -d "{\"question\": \"How do I install it?\", \"session_id\": \"$SID\"}"

# Clear the session
curl -X DELETE http://localhost:8000/session/$SID
```

---

## Auth per user

The server respects your `raggity.toml` auth configuration. For Claude, set up auth before starting the server:

```bash
claude login   # subscription
# or
export ANTHROPIC_API_KEY=sk-ant-...   # API key
```

### Multi-tenant (`server.per_user`)

With `server.auth = "api_key"` and `server.per_user = true`, each API key gets its own **isolated index namespace** and its own **session namespace** — one tenant can never read, continue, or delete another tenant's conversations or documents. Tenants ingest their own content via `POST /ingest/content`. Per-tenant `Raggity` instances are cached with an LRU bound (`server.max_user_rags`, default 128) and closed on eviction/shutdown.

#### Per-tenant personas

`server.personas` maps an API key to a persona string that is applied to that
tenant's `generation.persona` before its `Raggity` is constructed — so each
tenant's answers are personalized to them, while grounding/citation rules still
apply. Keys without an entry get the default (persona-free) system prompt.

```toml
[server]
auth = "api_key"
per_user = true
api_keys = ["key_alice", "key_bob"]

[server.personas]
key_alice = "The user is Alice, a maritime lawyer. Prefer precise legal phrasing."
```

---

## Terminal chat REPL

For interactive multi-turn conversation in the terminal without the HTTP server:

```bash
rag chat
```

Type a question and press Enter; raggity streams the answer token-by-token with verified source footnotes. Type `exit` or press Ctrl-D to quit.
