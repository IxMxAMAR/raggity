# Quickstart

## 1. Install

```bash
pip install raggity
```

## 2. Generate a config

```bash
rag init
```

Open the generated `raggity.toml` and point `sources.include` at your notes:

```toml
[sources]
include = ["~/notes/**/*.md", "~/docs/**/*.pdf"]
```

## 3. Index your sources

```bash
rag ingest
```

Ingestion is **incremental** — hash-based, so only new or changed files are processed. Safe to re-run at any time.

## 4. Ask a question

```bash
rag ask "How do I set up a new dev environment?"
```

raggity retrieves the most relevant chunks from your index, reranks them, and answers with verified inline citations. If no chunk clears the relevance threshold, it returns "I don't have enough information" rather than guessing.

---

## CLI reference

| Command | Description |
|---|---|
| `rag init` | Create a `raggity.toml` config template |
| `rag ingest` | Incrementally index configured sources |
| `rag ingest-url <url>` | Fetch a web URL (and optionally crawl same-domain links) |
| `rag ingest-repo <url>` | Shallow-clone a git repo and index all text files |
| `rag ingest-obsidian <vault>` | Index all Markdown notes from an Obsidian vault |
| `rag ask "..."` | Ask a question; prints the answer with verified source footnotes |
| `rag ask "..." --plain` | Pipe-friendly output — no Rich formatting, no footnotes |
| `rag ask "..." --hyde` | HyDE query transform — improves dense recall |
| `rag ask "..." --step-back` | Step-back query transform — higher-level context retrieval |
| `rag ask "..." --expand` | Multi-query expansion via RRF |
| `rag ask "..." --decompose` | Decompose into sub-questions, retrieve independently, merge |
| `rag ask "..." --no-cache` | Bypass the answer cache |
| `rag chat` | Interactive multi-turn chat REPL in the terminal |
| `rag serve` | Start the local HTTP API server |
| `rag serve --open` | Start the server and open the web chat UI |
| `rag status` | Show index statistics (chunk count, source count, index path) |
| `rag reindex --force` | Wipe and rebuild the index from scratch |
| `rag eval golden.jsonl` | Run retrieval quality metrics (Hit@k, MRR, Recall@k) |
| `rag eval golden.jsonl --llm-judge` | LLM-judge eval: faithfulness + answer relevance |
| `rag watch` | Watch source folders and re-index automatically on file changes |
| `rag graph-build` | Extract entities/relations and save `graph.json` |

All commands accept `--config PATH` to point at a non-default config file.

---

## Next steps

- [Configuration reference](configuration.md) — all `raggity.toml` knobs
- [Retrieval pipeline](retrieval.md) — tuning hybrid search, reranking, abstention
- [Ingestion](ingestion.md) — connectors, file types, OCR
- [Server & API](server.md) — HTTP server, sessions, SSE streaming
