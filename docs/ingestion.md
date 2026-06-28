# Ingestion

raggity indexes your content with `rag ingest`. Ingestion is **incremental** — hash-based, so only new or changed files are processed on subsequent runs.

---

## Supported file types

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

### Document extra

Install optional document readers:

```bash
pip install raggity[docs]
```

Adds support for `.docx`, `.html`, and `.pptx` files.

### OCR extra

For scanned PDFs and image files:

```bash
pip install raggity[ocr]
```

Adds RapidOCR + pypdfium2. raggity automatically OCRs a PDF page when embedded text is absent.

---

## Local file ingestion

Configure glob patterns in `raggity.toml`:

```toml
[sources]
include = [
  "~/notes/**/*.md",
  "~/docs/**/*.pdf",
  "~/projects/**/*.txt",
]
```

Run ingestion:

```bash
rag ingest
```

Force a full rebuild from scratch:

```bash
rag reindex --force
```

---

## Connectors

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

Configure URLs for automatic ingestion on every `rag ingest` run:

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

All text files with supported extensions are read and indexed. The index `path` for each document is `<repo_url>#<relpath>` so you can trace sources back to the repository.

### Obsidian vault (`rag ingest-obsidian`)

No extra install needed.

```bash
rag ingest-obsidian ~/Documents/MyVault
```

raggity walks all `.md` files in the vault recursively and normalises `[[wikilink]]` / `[[link|alias]]` syntax to plain text before indexing, so bracket noise does not pollute your chunks.

---

## Watch daemon

Auto-reindex on file changes. Install the extra:

```bash
pip install raggity[watch]
```

Start the daemon:

```bash
rag watch

# Customise debounce delay (default 2 s)
rag watch --debounce 5.0
```

raggity monitors all paths in `sources.include` recursively. When files change, it triggers a debounced re-index — rapid filesystem events are coalesced into a single call. The daemon runs until you press Ctrl-C.

---

## Chunking

raggity chunks documents into fixed-size pieces (default **256 tokens**) for indexing. When `retrieval.parent_document = true`, each chunk retains a reference to its parent document (up to 1024 tokens), which is used during retrieval to pass broader context to the LLM.

---

## Batch and parallel embedding

raggity automatically batches embedding calls and supports parallel workers:

```toml
[embedding]
batch_size = 256   # increase for faster ingest on large corpora
parallel = 0       # number of parallel workers (0 = auto)
```

### Embedding cache

Avoid re-embedding unchanged chunks across ingest runs:

```toml
[embedding]
cache = true   # cache embeddings as JSON under the index directory
```

Cached embeddings are stored at `<index.path>/embed_cache.json` and looked up by content hash before calling the embedding model. Useful for large corpora with small diffs.

---

## ANN auto-index

raggity automatically builds an Approximate Nearest Neighbor index on the vector store once the collection grows past a threshold (default 50 000 chunks):

```toml
[index]
ann_threshold = 50000
```

This keeps search latency flat as your knowledge base scales.
