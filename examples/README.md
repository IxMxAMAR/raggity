# raggity — example recipes

Three copy-paste recipes to get started quickly.

---

## Recipe 1: Personal notes knowledge base

Index your Markdown notes and PDFs, answered by Claude (default backend).

**Install**

```bash
pip install raggity
```

**`raggity.toml`**

```toml
[sources]
include = ["~/notes/**/*.md", "~/Documents/**/*.pdf"]

[index]
path = "~/.raggity/notes"

[generation]
backend = "claude"
model   = "claude-opus-4-8"
auth    = "auto"          # uses claude login session or ANTHROPIC_API_KEY
```

**Commands**

```bash
# Log in once (subscription — no API key needed)
claude login

# Index your notes (incremental — safe to re-run)
rag ingest

# Ask a question
rag ask "What did I write about async Python last week?"

# Start an interactive chat session
rag chat
```

Raggity only answers when it finds supporting evidence — it will tell you when
it doesn't know rather than hallucinating.

---

## Recipe 2: Fully offline (Ollama + local fastembed)

No cloud calls, no API key. Everything runs locally.

**Install**

```bash
pip install "raggity[openai]"
ollama pull llama3.1
```

**`raggity.toml`**

```toml
[sources]
include = ["~/notes/**/*.md"]

[index]
path = "~/.raggity/offline"

[embedding]
# fastembed runs locally on CPU — no change needed; this is the default

[generation]
backend  = "ollama"
model    = "llama3.1"
# base_url defaults to http://localhost:11434/v1
```

**Commands**

```bash
# Ollama must be running in the background
ollama serve &

rag ingest
rag ask "Summarise my meeting notes from Q2"
```

Embeddings run via ONNX Runtime on CPU (no GPU required). The Ollama server
handles generation locally. Zero data leaves your machine.

---

## Recipe 3: Web and GitHub knowledge base

Ingest documentation websites and a GitHub repository, then query them together.

**Install**

```bash
pip install "raggity[web]"
```

**`raggity.toml`**

```toml
[sources]
# Static URLs ingested on every `rag ingest`
urls = [
  "https://docs.anthropic.com/en/docs/overview",
]

[index]
path = "~/.raggity/web-kb"

[generation]
backend = "claude"
model   = "claude-opus-4-8"
auth    = "auto"
```

**Commands**

```bash
# One-off: crawl a docs site 2 hops deep and add to the index
rag ingest-url https://docs.anthropic.com/en/docs/overview --depth 2

# One-off: shallow-clone a GitHub repo and index all text files
rag ingest-repo https://github.com/IxMxAMAR/raggity

# Run regular ingestion (picks up the urls list from config)
rag ingest

rag ask "How does raggity handle hybrid retrieval?"
rag ask "What are the rate limits for the Claude API?"
```

Each document's source URL or `<repo>#<filepath>` path is tracked — citations
in the answer tell you exactly which page the information came from.
