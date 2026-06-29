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
data: Hello

data:  world

event: done
data: {"citations": ["source.md"], "session_id": "abc123"}
```

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

---

## Terminal chat REPL

For interactive multi-turn conversation in the terminal without the HTTP server:

```bash
rag chat
```

Type a question and press Enter; raggity streams the answer token-by-token with verified source footnotes. Type `exit` or press Ctrl-D to quit.
