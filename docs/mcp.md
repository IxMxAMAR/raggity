# MCP server

`rag mcp` exposes your knowledge base as a
[Model Context Protocol](https://modelcontextprotocol.io) server over **stdio**,
so MCP-aware agents — Claude Code, Claude Desktop, Cursor, and others — can query
your notes and documents as a native tool.

It speaks MCP on stdin/stdout; there is no HTTP port and nothing to expose to the
network. The agent launches `rag mcp` as a subprocess and talks to it directly.

## Install

The MCP server needs one extra dependency (the official MCP Python SDK):

```bash
pip install "raggity[mcp]"
```

Then make sure your index is built (`rag ingest`) — the server queries the
existing index; it never rebuilds it.

## Run

```bash
rag mcp --config C:/path/to/raggity.toml
```

The process blocks and serves MCP over stdio until the client disconnects. It
writes nothing to stdout except the JSON-RPC protocol traffic (logs go to
stderr), so the pipe stays clean.

## Register with Claude Code

```bash
claude mcp add raggity -- rag mcp --config C:/path/to/raggity.toml
```

Claude Code will spawn `rag mcp` on demand and list its tools. Omit `--config`
to use the `raggity.toml` discovered from the current directory.

## Register with Claude Desktop / Cursor

Add an entry to the MCP client's server config JSON (for Claude Desktop:
`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "raggity": {
      "command": "rag",
      "args": ["mcp", "--config", "C:/path/to/raggity.toml"]
    }
  }
}
```

Use an absolute path to `rag` (e.g. the one inside your virtualenv's
`Scripts`/`bin`) if it is not on the client's `PATH`.

## Tools

| Tool | What it does | Abstention |
|------|--------------|------------|
| `search(query, k=8, max_context_tokens=None)` | Retrieval only — no LLM. Returns the top-`k` chunks packed into one context string, each as a `[source: <path>]` block. Raw material for the **agent's own** reasoning. | **Bypassed** — always returns the best available matches; the agent decides relevance. |
| `ask(question)` | The full raggity RAG pipeline (retrieve → generate → verify citations). Returns the answer text plus a `Sources:` footer listing cited source paths. | **Kept** — if the knowledge base lacks enough information, raggity's abstention message is returned verbatim (no fabricated answer). |
| `kb_status()` | Cheap store-only summary: indexed chunk count, distinct source count, and index backend. Does not build the embedder, reranker, or LLM provider. | n/a |

### When to use `search` vs `ask`

- Use **`search`** when the agent wants grounding passages to reason over,
  quote, or cross-check itself. It bypasses the relevance floor, so the agent
  always sees the closest matches and makes its own call on sufficiency.
- Use **`ask`** when you want a finished, cited answer produced by the knowledge
  base's own pipeline — including its refusal to answer when the KB does not
  cover the question.
