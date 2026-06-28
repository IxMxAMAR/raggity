# raggity

[![CI](https://github.com/IxMxAMAR/raggity/actions/workflows/tests.yml/badge.svg)](https://github.com/IxMxAMAR/raggity/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/raggity.svg)](https://pypi.org/project/raggity/)
[![Python](https://img.shields.io/pypi/pyversions/raggity.svg)](https://pypi.org/project/raggity/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://github.com/IxMxAMAR/raggity/blob/main/LICENSE)

**Local-first, top-tier RAG over your notes, docs, and PDFs — answered by Claude.**

Hybrid retrieval (dense + BM25 + RRF), cross-encoder reranking, dedup, verified inline citations, and selective abstention: raggity only answers when it has evidence.

---

## What makes raggity different

| Feature | raggity |
|---|---|
| **Hybrid retrieval** | Dense vector + BM25 full-text, fused with Reciprocal Rank Fusion (RRF k=60) |
| **Cross-encoder reranking** | Local ONNX cross-encoder re-scores every candidate — no blind top-k |
| **Selective abstention** | Returns "I don't have enough information" instead of hallucinating |
| **Verified citations** | Inline citation markers are cross-checked against retrieved sources before display |
| **Zero-GPU default** | CPU-only ONNX Runtime — works on any machine |
| **Three LLM backends** | Claude (default), OpenAI-compatible APIs, Ollama (offline) |
| **Two vector stores** | LanceDB (local, zero-config) or Qdrant (scalable, multi-user) |
| **Full pipeline** | Query transforms, parent-doc retrieval, GraphRAG, semantic answer cache, SSE server |

---

## Architecture overview

```
Sources → Chunker → Embedder → LanceDB / Qdrant
                                      |
          Query ──────────────────────┤
                      dense search    ├── RRF fusion (k=60)
                      BM25/FTS        ┘         |
                                       Cross-encoder rerank
                                               |
                                         Dedup (cosine ≥ 0.92)
                                               |
                                       Relevance floor filter
                                               |
                                  Lost-in-the-middle reorder
                                               |
                               Claude Agent SDK → Answer
                               (with verified citations)
```

---

## Quick links

- [Installation](install.md)
- [Quickstart](quickstart.md)
- [Configuration reference](configuration.md)
- [Backends — LLM + vector stores](backends.md)
- [Retrieval pipeline](retrieval.md)
- [Ingestion — file types & connectors](ingestion.md)
- [Server & API](server.md)
- [Deploy — Docker & observability](deploy.md)

---

## License

**GNU AGPL-3.0-or-later.** If you modify raggity and distribute it — or run a modified version as a hosted service — you must release your source under the AGPL as well. Using raggity as-is to query your own documents has no such obligation.
