# raggity Launch Post Drafts

Ready-to-post copy for launch day. Edit links/stats as needed before posting.

---

## 1. Show HN

**Title:**
> Show HN: raggity – local-first RAG with Claude (no API key needed), hybrid retrieval + verified citations

**Body:**

Hi HN,

I built raggity (https://github.com/IxMxAMAR/raggity), a local-first RAG engine that lets you ask questions over your notes, PDFs, and docs — and only answers when it has evidence (selective abstention).

**What makes it different:**

- **No API key required by default.** raggity uses the Claude Agent SDK, so if you have a Claude subscription (`claude login`), you're already good. API key, OpenAI-compatible endpoints, and offline Ollama are all supported too.
- **CPU-first, torch-free, cross-vendor.** Embeddings and reranking run on ONNX Runtime — no CUDA setup, no GPU needed. Works on AMD, NVIDIA, and no-GPU machines equally. Works on macOS, Linux, and Windows.
- **Batteries-included retrieval pipeline.** Hybrid dense+BM25 fusion (RRF), cross-encoder reranking, deduplication, parent-document expansion, lost-in-the-middle reorder, and a relevance floor that triggers abstention when nothing is confident enough.
- **Verified inline citations.** Every answer lists which source chunks it drew from. The citation markers are cross-checked against retrieved chunks — hallucinated source numbers are dropped.
- **Local-first.** LanceDB (no server needed) by default. Switch to Qdrant for multi-user/large-scale.
- **Multi-format ingest.** Markdown, PDF, DOCX, PPTX, HTML, CSV, images (OCR), web URLs, Git repos, Obsidian vaults — one `rag ingest` covers them all.
- **Optional extras:** web chat UI, FastAPI server, GraphRAG, Docker + Qdrant compose, file-system watch daemon, OpenTelemetry, LLM-judge evals.

```bash
pip install raggity
rag ingest          # index ~/notes and ~/docs per raggity.toml
rag ask "..."       # cited answer, or "I don't have enough information…"
```

License: AGPL-3.0. PyPI: https://pypi.org/project/raggity/

Happy to answer questions about the retrieval design or the abstention logic.

---

## 2. r/LocalLLaMA and r/selfhosted

**Title:**
> raggity: local-first RAG that works with your Claude subscription (no API key), or Ollama offline — hybrid retrieval, reranking, verified citations, selective abstention

**Body:**

Hey everyone,

Just launched **raggity** — a local-first RAG CLI/server that I've been building to be actually practical for personal knowledge bases.

**The pitch in 30 seconds:**

```bash
pip install raggity
claude login       # <-- already have Claude? you're done, no API key needed
rag ingest         # index your notes, PDFs, docs
rag ask "How do I configure X?"
# → cited answer, or "I don't have enough information" if nothing is relevant
```

**Key differentiators vs. other local RAG tools:**

| Feature | raggity |
|---|---|
| Claude subscription (no API key) | Yes — via Claude Agent SDK |
| OpenAI-compatible / Ollama offline | Yes (same install, just change `backend =`) |
| GPU required | No — ONNX Runtime CPU, works everywhere |
| AMD GPU support | Yes (DirectML on Windows, ROCm on Linux) |
| Retrieval | Hybrid dense+BM25, RRF fusion, cross-encoder rerank |
| Selective abstention | Yes — drops below relevance floor, no hallucinated confidence |
| Verified citations | Yes — markers cross-checked against retrieved chunks |
| Parent-document retrieval | Yes |
| GraphRAG | Yes (opt-in) |
| Web chat UI + API server | Yes (`rag serve --open`) |
| Docker + Qdrant | Yes |
| OpenTelemetry | Yes |
| License | AGPL-3.0 |

**Retrieval pipeline (simplified):**

```
Query → dense + BM25 → RRF fusion → cross-encoder rerank
      → relevance floor → abstain OR send to Claude with verified citations
```

**Supported sources:** Markdown, PDF (text + OCR fallback), DOCX, PPTX, HTML, CSV, images, web URLs (with crawl), Git repos, Obsidian vaults.

- GitHub: https://github.com/IxMxAMAR/raggity
- PyPI: https://pypi.org/project/raggity/
- Docs: https://ixmxamar.github.io/raggity/

Would love feedback on the retrieval design, especially if you have a large knowledge base to throw at it. AMA.

---

## 3. X / Twitter Thread

**Tweet 1:**
Launching raggity — local-first RAG over your notes & docs, answered by Claude.

No API key needed if you have a Claude subscription. AGPL-3.0, batteries included.

pip install raggity
rag ingest && rag ask "…"

🔗 https://github.com/IxMxAMAR/raggity

[thread 🧵]

---

**Tweet 2:**
Why build another RAG tool?

Most tools either:
• require an API key + dollars per query
• hallucinate citations
• fail gracefully when they have no evidence

raggity does none of those things.

---

**Tweet 3:**
Works with your Claude subscription via the Agent SDK — no API key.
Switch to OpenAI-compatible (GPT, Groq, Together…) or local Ollama in one config line.

```toml
[generation]
backend = "ollama"
model   = "llama3.1"
```

---

**Tweet 4:**
CPU-first, torch-free.

Embeddings + reranking run on ONNX Runtime. No CUDA, no GPU required.

AMD? Works (DirectML / ROCm). NVIDIA? Works. No GPU at all? Works.

---

**Tweet 5:**
Retrieval pipeline:

dense vector search + BM25 full-text → RRF fusion
→ cross-encoder rerank
→ relevance floor (sigmoid-normalised score)
→ if nothing clears the floor: "I don't have enough information"

No false confidence. Abstention is a first-class citizen.

---

**Tweet 6:**
Citations are verified, not assumed.

After generation, each inline [1] [2] marker is cross-checked against retrieved chunk IDs. Hallucinated source numbers are dropped. You see only citations that actually exist in your index.

---

**Tweet 7:**
Ingest from:
• Markdown / TXT / PDF / DOCX / PPTX / HTML / CSV
• Images (OCR via RapidOCR)
• Web URLs (crawl same-domain links)
• Git repos (shallow clone)
• Obsidian vaults (wikilink normalization)

One `rag ingest` covers everything.

---

**Tweet 8:**
Also ships:
• Web chat UI (`rag serve --open`)
• FastAPI REST + SSE streaming
• GraphRAG (entity graph over your chunks)
• Docker + Qdrant compose
• File-system watch daemon (`rag watch`)
• LLM-judge evals
• OpenTelemetry

---

**Tweet 9:**
It's AGPL-3.0 and lives here:

GitHub: https://github.com/IxMxAMAR/raggity
PyPI:   https://pypi.org/project/raggity/
Docs:   https://ixmxamar.github.io/raggity/

Feedback, issues, and PRs very welcome. What would you ingest first?
