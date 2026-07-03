"""GraphRAG: entity/relation extraction + NetworkX-backed GraphStore.

Opt-in via the ``raggity[graph]`` extra (``networkx>=3.1``).
"""
from __future__ import annotations

import json
import re
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import networkx as _nx_type

# ---------------------------------------------------------------------------
# System prompt used for extraction
# ---------------------------------------------------------------------------
_EXTRACT_SYSTEM = (
    "You are a knowledge-graph extractor. "
    "For each piece of text you receive, output ONLY lines in these two formats:\n"
    "  E: <entity name>\n"
    "  R: <source entity> | <relation> | <target entity>\n"
    "Do not output any other text, explanations, or commentary. "
    "Omit lines for things you are not confident about."
)

_EXTRACT_PROMPT_TMPL = "Extract entities and relations from this text:\n\n{text}"


# ---------------------------------------------------------------------------
# extract()
# ---------------------------------------------------------------------------

async def extract(
    chunk_text: str,
    provider,
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Call *provider* once and parse ``E:`` / ``R:`` lines.

    Returns
    -------
    entities
        Deduplicated display labels (original casing, case-insensitive dedup).
    relations
        List of ``(src_label, relation, dst_label)`` 3-tuples, display casing.
    """
    prompt = _EXTRACT_PROMPT_TMPL.format(text=chunk_text)
    raw: str = await provider.complete(_EXTRACT_SYSTEM, prompt)

    seen_entity_keys: set[str] = set()
    entities: list[str] = []
    relations: list[tuple[str, str, str]] = []

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("E:"):
            label = line[2:].strip()
            if not label:
                continue
            key = label.lower()
            if key not in seen_entity_keys:
                seen_entity_keys.add(key)
                entities.append(label)
        elif line.startswith("R:"):
            parts = [p.strip() for p in line[2:].split("|")]
            if len(parts) != 3:
                # Malformed – skip (too few or too many pipe segments)
                continue
            src, rel, dst = parts
            if not (src and rel and dst):
                continue
            relations.append((src, rel, dst))
        # Any other line is silently ignored (garbage / prose)

    return entities, relations


# ---------------------------------------------------------------------------
# GraphStore
# ---------------------------------------------------------------------------

class GraphStore:
    """In-memory directed graph of entities and relations backed by networkx.

    Node key  = entity label lowercased.
    Node attrs = ``{label: str, chunk_ids: set[str]}``.
    Edge attrs = ``{relation: str}``.

    Raises ``RuntimeError`` with an install hint if networkx is missing.
    """

    def __init__(self) -> None:
        try:
            import networkx as nx  # lazy import – only when class is used
        except ImportError:
            raise RuntimeError(
                "pip install raggity[graph]"
            ) from None
        self._nx = nx
        self._graph: _nx_type.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(
        self,
        entities: list[str],
        relations: list[tuple[str, str, str]],
        chunk_id: str,
    ) -> None:
        """Add entities + relations extracted from *chunk_id* to the graph."""
        g = self._graph

        for label in entities:
            key = label.lower()
            if key in g:
                g.nodes[key]["chunk_ids"].add(chunk_id)
                # Keep first-seen label; do not overwrite
            else:
                g.add_node(key, label=label, chunk_ids={chunk_id})

        for src_label, rel, dst_label in relations:
            src_key = src_label.lower()
            dst_key = dst_label.lower()
            # Ensure nodes exist (relation may reference entities not in entity list)
            if src_key not in g:
                g.add_node(src_key, label=src_label, chunk_ids={chunk_id})
            if dst_key not in g:
                g.add_node(dst_key, label=dst_label, chunk_ids={chunk_id})
            g.add_edge(src_key, dst_key, relation=rel)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return number of nodes."""
        return self._graph.number_of_nodes()

    def link(self, query_entities: list[str]) -> set[str]:
        """Return the set of node keys that match any query entity.

        Matching is case-insensitive: exact match first, then word-boundary
        containment (query term appears as a whole word inside a node key).
        Raw substring match (e.g. 'ai' ∈ 'training') is intentionally excluded
        to avoid false positives.
        """
        matched: set[str] = set()
        node_keys = list(self._graph.nodes())
        for qe in query_entities:
            qe_lower = qe.lower()
            # Escape special regex chars in the query entity
            pattern = re.compile(r"\b" + re.escape(qe_lower) + r"\b")
            for nk in node_keys:
                if qe_lower == nk or pattern.search(nk):
                    matched.add(nk)
        return matched

    def neighborhood_chunk_ids(
        self,
        nodes: set[str],
        hops: int = 1,
    ) -> set[str]:
        """BFS up to *hops* hops over the undirected view; union chunk_ids.

        At ``hops=0`` only the seed nodes themselves are included.
        """
        g = self._graph
        visited: set[str] = set()
        # Treat the directed graph as undirected for neighbourhood expansion
        undirected = g.to_undirected(as_view=True)

        frontier: deque[tuple[str, int]] = deque()
        for n in nodes:
            if n in g:
                frontier.append((n, 0))
                visited.add(n)

        while frontier:
            node, depth = frontier.popleft()
            if depth < hops:
                for neighbour in undirected.neighbors(node):
                    if neighbour not in visited:
                        visited.add(neighbour)
                        frontier.append((neighbour, depth + 1))

        chunk_ids: set[str] = set()
        for n in visited:
            chunk_ids.update(g.nodes[n].get("chunk_ids", set()))
        return chunk_ids

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the graph to a JSON node-link file at *path*.

        ``chunk_ids`` sets are stored as sorted lists so the file is
        deterministic and JSON-serialisable.
        """
        nx = self._nx
        data = nx.node_link_data(self._graph, edges="links")
        # Convert chunk_ids sets → lists for JSON
        for node in data.get("nodes", []):
            if "chunk_ids" in node and isinstance(node["chunk_ids"], set):
                node["chunk_ids"] = sorted(node["chunk_ids"])
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """Load a graph previously saved with :meth:`save`, replacing current graph."""
        nx = self._nx
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        # Restore chunk_ids lists → sets
        for node in data.get("nodes", []):
            if "chunk_ids" in node and isinstance(node["chunk_ids"], list):
                node["chunk_ids"] = set(node["chunk_ids"])
        self._graph = nx.node_link_graph(data, directed=True, edges="links")


# ---------------------------------------------------------------------------
# build_graph()
# ---------------------------------------------------------------------------

async def build_graph(chunks, provider, concurrency: int = 8) -> GraphStore:
    """Extract entities/relations from each chunk and accumulate in a GraphStore.

    Extraction runs concurrently (bounded by *concurrency*) but nodes are added
    in ORIGINAL chunk order so the resulting graph is deterministic — the
    first-seen-label rule in :meth:`GraphStore.add` is order-sensitive.  A chunk
    whose extraction raises contributes nothing (``([], [])``) rather than
    aborting the whole build.

    Parameters
    ----------
    chunks
        Iterable of objects with ``.text`` and ``.chunk_id`` attributes.
    provider
        An :class:`~raggity.llm.LLMProvider` instance.
    concurrency
        Maximum number of in-flight extraction calls.
    """
    import asyncio

    chunk_list = list(chunks)
    store = GraphStore()
    if not chunk_list:
        return store
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(chunk):
        async with sem:
            try:
                return await extract(chunk.text, provider)
            except Exception:
                return ([], [])

    results = await asyncio.gather(*[_one(c) for c in chunk_list])
    # Add in original order for determinism (gather preserves input order).
    for chunk, (entities, relations) in zip(chunk_list, results):
        store.add(entities, relations, chunk.chunk_id)
    return store
