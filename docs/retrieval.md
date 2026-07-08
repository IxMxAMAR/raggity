# Retrieval

raggity's retrieval pipeline combines hybrid search, cross-encoder reranking, and several advanced features to maximise both precision and recall.

---

## Pipeline overview

```
Query
  │
  ├─── Dense vector search  ──┐
  │                            ├── RRF fusion (k=60)
  └─── BM25 / FTS search   ──┘        │
                                 Cross-encoder rerank
                                       │
                                 Dedup (cosine ≥ 0.92)
                                       │
                          Optional rerank-score filter
                                       │
                             Lost-in-the-middle reorder
                                       │
                              LLM → Answer (with citations)
```

---

## Hybrid retrieval + RRF fusion

raggity runs two parallel retrievers on every query:

1. **Dense vector search** — semantic similarity via embeddings.
2. **BM25 / full-text search** — keyword precision.

The two ranked lists are merged with **Reciprocal Rank Fusion** (RRF, k=60). RRF is robust — it does not require score normalisation and handles the two very different score distributions gracefully.

```toml
[retrieval]
candidates = 30   # candidates fetched from each retriever before fusion
rrf_k = 60        # RRF constant (higher = flatter fusion curve)
```

---

## Cross-encoder reranking

Enabled by default. After RRF fusion, a local ONNX cross-encoder scores every candidate chunk jointly with the query — a much stronger signal than bi-encoder similarity alone.

```toml
[retrieval]
rerank = true
rerank_model = "Xenova/ms-marco-MiniLM-L-6-v2"   # default (fast)
# rerank_model = "BAAI/bge-reranker-v2-m3"        # heavier, higher quality
```

---

## Deduplication

Chunks with **cosine similarity ≥ `dedup_cosine`** (default 0.92) are collapsed — the highest-ranked copy is kept, duplicates are dropped.

```toml
[retrieval]
dedup_cosine = 0.92
```

---

## Selective abstention

raggity abstains — returning "I don't have enough information" without calling the LLM — when the **top dense cosine similarity** among retrieved candidates falls below `sufficiency_floor` (default 0.5), or when retrieval returns no candidates at all.

```toml
[retrieval]
sufficiency_floor = 0.5   # dense-cosine similarity threshold; below this, raggity abstains
```

`sufficiency_floor` is the primary abstention knob. Lower it to answer more questions (at the risk of lower precision); raise it to be more conservative.

The **cross-encoder reranker** affects **ordering only** — its absolute score does not govern abstention. ms-marco cross-encoders produce near-zero absolute scores even for correct matches on casual phrasings, making the raw score an unreliable abstention signal.

`relevance_floor` is an **optional secondary filter** on the sigmoid-normalised cross-encoder rerank score, **default 0.0 (off)**. When set above 0.0, chunks whose sigmoid-normalised rerank score falls below the threshold are dropped for ordering/trimming purposes. It does **not** itself trigger abstention if it empties the candidate list — only `sufficiency_floor` abstains.

---

## Lost-in-the-middle reorder

Top-scoring chunks are placed at the **start and end** of the context window where LLMs attend most strongly. Mid-ranked chunks fill the middle. This is transparent and always active.

---

## Parent-document retrieval

By default, raggity indexes 256-token chunks. When `parent_document = true`, each chunk retains a reference to its parent (up to 1024 tokens). Retrieval expands matched chunks to include their parents before passing context to the LLM — better for questions that need broader context.

```toml
[retrieval]
parent_document = true
```

!!! note
    Enabling or disabling `parent_document` triggers an automatic full index rebuild via the index fingerprint.

---

## Query transforms

Query transforms generate additional queries to improve retrieval coverage. They compose freely — all generated queries are fused via RRF before reranking.

### HyDE — Hypothetical Document Embeddings

Generates a **hypothetical answer passage** using the LLM, then uses that passage as an additional dense query vector. Useful when the answer vocabulary differs significantly from the question vocabulary.

```bash
rag ask "What are the main tradeoffs of eventual consistency?" --hyde
```

Enable permanently in config:

```toml
[retrieval]
hyde = true
```

### Step-back prompting

Generates a broader, more abstract "step-back" question, retrieves for it alongside the original, and merges the results. Useful for grounding specific questions in general principles.

```bash
rag ask "How do I configure the database connection pool?" --step-back
```

Enable permanently:

```toml
[retrieval]
step_back = true
```

!!! note
    Both `--hyde` and `--step-back` add one LLM call each and compose freely with `--expand`.

### Query expansion

Generates multiple query variations, retrieves for each, and reranks the combined results via RRF. Improves coverage for complex questions.

```bash
rag ask "How do I set up a new dev environment?" --expand
```

Configure the number of variations:

```toml
[retrieval]
expand_n = 3   # number of query variations (default 3)
```

### Decompose

Breaks a complex question into **sub-questions**, retrieves chunks for each independently, merges by chunk ID (dedup), and answers over the combined context. Useful for multi-faceted questions.

```bash
rag ask "How do backups, retention policies, and restore procedures interact?" --decompose
```

`--decompose` overrides `--hyde`, `--step-back`, and `--expand` when combined.

---

## GraphRAG (opt-in)

GraphRAG augments hybrid retrieval with a **knowledge graph** — entities and relations extracted from indexed chunks. At query time, entities mentioned in the question are linked to the graph and their neighbourhood chunks are merged into the candidate set before reranking.

**GraphRAG is off by default.** It is LLM-cost-heavy (one call per chunk to build the graph).

Enable in `raggity.toml`:

```toml
[retrieval]
graph = true
graph_hops = 1   # BFS hops from matched entities (default 1)
```

Install the extra:

```bash
pip install raggity[graph]
```

Build the graph:

```bash
rag graph-build   # one LLM call per indexed chunk
```

Or enable `retrieval.graph = true` in config and let `rag ingest` build it automatically.

The graph is saved to `<index.path>/graph.json` and loaded automatically on startup when `graph = true`.

!!! warning
    `rag graph-build` makes one LLM call per chunk. For large corpora, build once and rebuild only when content changes significantly.

---

## Corrective retrieval (CRAG-style)

Corrective retrieval adds a lightweight self-check after the first retrieval, following the CRAG pattern (a retrieval evaluator plus one corrective round). When enabled, raggity:

1. **Evaluates** the retrieved passages with one LLM call, grading them `correct`, `incorrect`, or `ambiguous` for the question.
2. On `incorrect`/`ambiguous`, runs **exactly one corrective round**: it rewrites the query toward the missing information (one LLM call), re-retrieves with `[original, rewritten]` (the original stays in the RRF fusion), then merges the new chunks with the first round (dedup by chunk id, rerank if enabled, reslice to `top_k`, lost-in-the-middle order).
3. On `correct`, proceeds unchanged.

If the first retrieval abstained (no chunks), the corrective round still gets its one rewrite-and-retrieve shot; if that also comes back empty, raggity abstains as usual.

**Corrective retrieval is off by default.** Enable in `raggity.toml`:

```toml
[retrieval]
corrective = true
```

**Cost:** +1 LLM call per question for the evaluator, and +1 more (the rewrite) whenever a corrective round is triggered. The evaluator verdict is cached in the answer cache (keyed on question + retrieved chunk ids + model) when `generation.cache = true`, so a repeat of the same question over the same retrieval skips the evaluator call. Evaluator or rewriter LLM failures log a warning and fall back to the original chunks — corrective never degrades below non-corrective behavior.

---

## Semantic answer cache

When enabled, raggity stores answers in `<index.path>/answer_cache.json`, keyed on SHA-256 of the question + retrieved chunk IDs + model name. Cache hits return immediately with no LLM call.

```toml
[generation]
cache = true
```

Bypass the cache for a single query:

```bash
rag ask "..." --no-cache
```

**Notes:**
- Cache is off by default.
- Cache hits are exact: same question + same chunks + same model.
- The streaming path always calls the model; only the buffered path reads/writes the cache.
- Delete `<index.path>/answer_cache.json` to clear the cache.

---

## Tuning reference

| Key | Default | Notes |
|---|---|---|
| `candidates` | `30` | Chunks fetched from each retriever before fusion |
| `top_k` | `5` | Chunks passed to the LLM after all filtering |
| `rerank` | `true` | Enable cross-encoder reranking |
| `sufficiency_floor` | `0.5` | Dense-cosine similarity threshold below which raggity abstains |
| `relevance_floor` | `0.0` | Optional secondary filter on the sigmoid-normalised cross-encoder rerank score (0.0 = off); does not trigger abstention |
| `hybrid` | `true` | Enable hybrid (dense + BM25) retrieval |
| `dedup_cosine` | `0.92` | Cosine threshold for dedup collapse |
| `rrf_k` | `60` | RRF constant |
| `parent_document` | `false` | Expand matched chunks to parent documents |
| `hyde` | `false` | Enable HyDE permanently |
| `step_back` | `false` | Enable step-back permanently |
| `expand_n` | `3` | Query variations for `--expand` |
| `graph` | `false` | Enable GraphRAG |
| `graph_hops` | `1` | BFS hops in the knowledge graph |
| `corrective` | `false` | Enable CRAG-style corrective retrieval (evaluator + one rewrite-and-merge round) |
