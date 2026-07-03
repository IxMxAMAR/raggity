import pytest
import raggity.llm as llm_mod
from raggity.config import RaggityConfig, SourcesConfig, IndexConfig


class _Block:
    def __init__(self, text): self.text = text


class _AssistantMessage:
    def __init__(self, text): self.content = [_Block(text)]


def test_core_ingest_and_status(tmp_path, monkeypatch):
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str(notes / "*.md")]),
        index=IndexConfig(path=str(tmp_path / "idx")),
    )
    from raggity.core import Raggity
    rag = Raggity(cfg)
    report = rag.ingest()
    assert report.added >= 1
    st = rag.status()
    assert st["chunks"] >= 1


def test_core_ask_uses_pipeline(tmp_path, monkeypatch):
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str(notes / "*.md")]),
        index=IndexConfig(path=str(tmp_path / "idx")),
    )

    async def _fake_query(prompt, options):
        yield _AssistantMessage("Backups run nightly to the NAS [doc_1#" +
                                "00000000].")

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    from raggity.core import Raggity
    rag = Raggity(cfg)
    rag.ingest()
    ans = rag.ask("how are backups done?")
    assert "NAS" in ans.text


def test_core_ask_hyde_routes_retrieve_multi(tmp_path, monkeypatch):
    """aask with hyde=True must call retrieve_multi (not retrieve) and return an answer."""
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str(notes / "*.md")]),
        index=IndexConfig(path=str(tmp_path / "idx")),
    )

    # Single mock handles both HyDE generation and final answer (via llm_mod.query)
    async def _fake_query(prompt, options):
        if "Question:" in prompt:
            yield _AssistantMessage("Backups are stored on a NAS device nightly.")
        else:
            yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    from raggity.core import Raggity
    rag = Raggity(cfg)
    rag.ingest()

    retrieve_multi_calls = []
    original_retrieve_multi = rag.retriever.retrieve_multi
    def _spy_multi(queries, question, **kwargs):
        retrieve_multi_calls.append(queries)
        return original_retrieve_multi(queries, question, **kwargs)
    monkeypatch.setattr(rag.retriever, "retrieve_multi", _spy_multi)

    ans = rag.ask("how are backups done?", hyde=True)
    assert "NAS" in ans.text
    assert len(retrieve_multi_calls) == 1, "retrieve_multi must be called when hyde=True"
    assert len(retrieve_multi_calls[0]) >= 2, "queries list must include original + hyde passage"


def test_aask_cache_hit_skips_model(tmp_path, monkeypatch):
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(sources=SourcesConfig(include=[str(notes / "*.md")]),
                        index=IndexConfig(path=str(tmp_path / "idx")))
    cfg.generation.cache = True
    calls = {"n": 0}
    async def _ans(prompt, options):
        calls["n"] += 1
        yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
    monkeypatch.setattr(llm_mod, "query", _ans)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)
    from raggity.core import Raggity
    rag = Raggity(cfg); rag.ingest()
    a1 = rag.ask("how are backups done?")
    a2 = rag.ask("how are backups done?")   # identical → cache hit, no 2nd model call
    assert a1.text == a2.text
    assert calls["n"] == 1


def test_core_qdrant_backend_ingest_ask(tmp_path, monkeypatch):
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(sources=SourcesConfig(include=[str(notes / "*.md")]),
                        index=IndexConfig(path=str(tmp_path / "idx"), backend="qdrant",
                                          qdrant_location=":memory:", qdrant_collection="t"))
    async def _fake_query(prompt, options):
        yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)
    from raggity.core import Raggity
    rag = Raggity(cfg); rag.ingest()
    assert rag.status()["chunks"] >= 1
    ans = rag.ask("how are backups done?")
    assert "NAS" in ans.text


def test_aask_decompose_merges_and_answers(tmp_path, monkeypatch):
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(sources=SourcesConfig(include=[str(notes / "*.md")]),
                        index=IndexConfig(path=str(tmp_path / "idx")))
    # All LLM calls (decompose + answer) go through llm_mod.query
    async def _fake(prompt, options):
        if "sub-questions" in prompt or "Question:" in prompt and "Give at most" in prompt:
            yield _AssistantMessage("how often?\nwhere stored?")
        else:
            yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
    monkeypatch.setattr(llm_mod, "query", _fake)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)
    from raggity.core import Raggity
    rag = Raggity(cfg); rag.ingest()
    import asyncio
    ans = asyncio.run(rag.aask_decompose("how are backups done?"))
    assert "NAS" in ans.text


def test_core_build_graph_requires_graph_true(tmp_path):
    """build_graph raises RuntimeError when cfg.retrieval.graph=False (default)."""
    from raggity.config import RaggityConfig, IndexConfig
    from raggity.core import Raggity
    import asyncio
    cfg = RaggityConfig(index=IndexConfig(path=str(tmp_path / "idx")))
    rag = Raggity(cfg)
    with pytest.raises(RuntimeError, match="graph"):
        asyncio.run(rag.build_graph())


async def test_core_build_graph_saves_graph_json(tmp_path, monkeypatch):
    """build_graph creates graph.json after extraction (mocked LLM)."""
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    from raggity.config import RaggityConfig, SourcesConfig, IndexConfig, RetrievalConfig
    from raggity.core import Raggity
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str(notes / "*.md")]),
        index=IndexConfig(path=str(tmp_path / "idx")),
        retrieval=RetrievalConfig(graph=True),
    )

    class _Block:
        def __init__(self, t): self.text = t
    class _AM:
        def __init__(self, t): self.content = [_Block(t)]

    async def _fake_query(prompt, options):
        yield _AM("E: NAS\nE: Backup System\nR: Backup System | writes to | NAS\n")

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AM)

    rag = Raggity(cfg)
    rag.ingest()
    # graph.json must exist after ingest with graph=true
    import os
    assert os.path.isfile(str(tmp_path / "idx" / "graph.json"))
    # rag._graph must be loaded
    assert rag._graph is not None
    assert rag._graph.count() >= 1


async def test_core_build_graph_standalone(tmp_path, monkeypatch):
    """Explicit rag graph-build path: ingest first (no graph), then build_graph()."""
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    from raggity.config import RaggityConfig, SourcesConfig, IndexConfig, RetrievalConfig
    from raggity.core import Raggity
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str(notes / "*.md")]),
        index=IndexConfig(path=str(tmp_path / "idx")),
        retrieval=RetrievalConfig(graph=True),
    )

    class _Block:
        def __init__(self, t): self.text = t
    class _AM:
        def __init__(self, t): self.content = [_Block(t)]

    async def _fake_query(prompt, options):
        yield _AM("E: NAS\nE: Backups\n")

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AM)

    rag = Raggity(cfg)
    # Manually ingest without graph building by temporarily disabling graph
    from raggity.config import RetrievalConfig as RC
    import raggity.core as core_mod
    # Ingest without triggering graph build: just call indexer directly
    from raggity.indexer import Indexer
    chunk_kwargs = {"parent_document": False, "parent_target_tokens": 1024,
                    "child_target_tokens": 256}
    indexer = Indexer(rag.embedder, rag.store, rag._manifest_path(),
                      fingerprint=rag._fingerprint(), chunk_kwargs=chunk_kwargs,
                      ann_threshold=rag.cfg.index.ann_threshold)
    indexer.ingest(cfg.sources.include)
    assert rag.store.count() >= 1

    await rag.build_graph()
    import os
    assert os.path.isfile(str(tmp_path / "idx" / "graph.json"))
    assert rag._graph is not None and rag._graph.count() >= 1


def test_core_graph_load_on_init(tmp_path, monkeypatch):
    """Raggity loads graph.json on __init__ when cfg.retrieval.graph=True and file exists."""
    from raggity.config import RaggityConfig, IndexConfig, RetrievalConfig
    from raggity.core import Raggity
    from raggity.graph import GraphStore

    # Write a fake graph.json
    idx = tmp_path / "idx"; idx.mkdir()
    g = GraphStore()
    g.add(["NAS"], [], "c1")
    g.save(str(idx / "graph.json"))

    cfg = RaggityConfig(index=IndexConfig(path=str(idx)),
                        retrieval=RetrievalConfig(graph=True))
    rag = Raggity(cfg)
    assert rag._graph is not None
    assert rag._graph.count() == 1


def test_fingerprint_changes_with_chunk_params(tmp_path):
    """_fingerprint() must change when chunk parameters change so stale chunks are evicted."""
    from raggity.config import RaggityConfig, IndexConfig, RetrievalConfig
    from raggity.core import Raggity

    cfg1 = RaggityConfig(
        index=IndexConfig(path=str(tmp_path / "idx")),
        retrieval=RetrievalConfig(parent_document=False, parent_target_tokens=1024,
                                  child_target_tokens=256),
    )
    cfg2 = RaggityConfig(
        index=IndexConfig(path=str(tmp_path / "idx")),
        retrieval=RetrievalConfig(parent_document=False, parent_target_tokens=2048,
                                  child_target_tokens=512),
    )
    cfg3 = RaggityConfig(
        index=IndexConfig(path=str(tmp_path / "idx")),
        retrieval=RetrievalConfig(parent_document=True, parent_target_tokens=1024,
                                  child_target_tokens=256),
    )

    rag1 = Raggity(cfg1)
    rag2 = Raggity(cfg2)
    rag3 = Raggity(cfg3)

    fp1 = rag1._fingerprint()
    fp2 = rag2._fingerprint()
    fp3 = rag3._fingerprint()

    assert fp1 != fp2, "different parent/child token targets → different fingerprint"
    assert fp1 != fp3, "parent_document=True vs False → different fingerprint"
    assert fp2 != fp3, "all three must be distinct"


async def test_aask_decompose_applies_ordering(tmp_path, monkeypatch):
    """aask_decompose must apply order_lost_in_middle on the merged pool.

    We verify ordering by patching retriever.retrieve to return chunks with known
    scores and checking that the best-scored chunk is at the head of what the
    answerer receives."""
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str(notes / "*.md")]),
        index=IndexConfig(path=str(tmp_path / "idx")),
    )

    class _Block:
        def __init__(self, t): self.text = t
    class _AM:
        def __init__(self, t): self.content = [_Block(t)]

    async def _fake_query(prompt, options):
        if "sub-questions" in prompt or ("Question:" in prompt and "Give at most" in prompt):
            yield _AM("how often?\nwhere stored?")
        else:
            yield _AM("Backups run nightly to the NAS [doc_1#00000000].")

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AM)

    from raggity.core import Raggity
    from raggity.models import Chunk
    rag = Raggity(cfg)
    rag.ingest()

    # Patch retriever.retrieve to return known chunks with scores.
    # Design: first call returns c_low first (score=0.3) then c_high (score=0.9)
    # so insertion order in merged dict is [c_low, c_high, c_med].
    # Without ordering fix: answerer gets [c_low(0.3), c_high(0.9), c_med(0.6)] — c_low at head
    # With fix (order_lost_in_middle): c_high(0.9) at head
    call_count = [0]
    def _fake_retrieve(q):
        call_count[0] += 1
        if call_count[0] == 1:
            # NOTE: c_low first, then c_high — so without ordering, c_low is at head
            c_low = Chunk("low-score text", "a.md", "A", "A", 1, "c_low", score=0.3)
            c_high = Chunk("high-score text", "a.md", "A", "A", 0, "c_high", score=0.9)
            return [c_low, c_high]
        else:
            c_med = Chunk("medium text", "a.md", "A", "A", 2, "c_med", score=0.6)
            return [c_med]
    monkeypatch.setattr(rag.retriever, "retrieve", _fake_retrieve)

    # Spy on answerer.answer to capture the chunks
    chunks_to_answer = []
    original_answer = rag.answerer.answer
    async def _spy_answer(question, chunks, **kw):
        chunks_to_answer.append(list(chunks))
        return await original_answer(question, chunks, **kw)
    monkeypatch.setattr(rag.answerer, "answer", _spy_answer)

    ans = await rag.aask_decompose("how are backups done?")
    assert "NAS" in ans.text
    assert len(chunks_to_answer) == 1

    chunks = chunks_to_answer[0]
    assert len(chunks) >= 2, "merged pool must have multiple chunks"

    # After fix: order_lost_in_middle applied → best score at head or tail
    # c_high (0.9) must be at an edge
    scores = [c.score for c in chunks]
    edge_scores = {scores[0], scores[-1]}
    max_score = max(scores)
    assert max_score in edge_scores, (
        f"Best score {max_score} must be at edge (head or tail); scores={scores}"
    )


async def test_transform_failure_falls_back_to_base_query(tmp_path, monkeypatch):
    """If a query transform (hyde/expand/step_back) raises, aask must still return
    an answer via base retrieval, not propagate the exception."""
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str(notes / "*.md")]),
        index=IndexConfig(path=str(tmp_path / "idx")),
    )

    class _Block:
        def __init__(self, t): self.text = t
    class _AM:
        def __init__(self, t): self.content = [_Block(t)]

    async def _fake_query(prompt, options):
        yield _AM("Backups run nightly to the NAS [doc_1#00000000].")

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AM)

    # Patch the HyDE generator to raise
    import raggity.query_transform as qt_mod
    async def _bad_hyde(question, provider):
        raise RuntimeError("HyDE LLM failed")
    monkeypatch.setattr(qt_mod, "generate_hyde_document", _bad_hyde)

    from raggity.core import Raggity
    rag = Raggity(cfg)
    rag.ingest()

    # With hyde=True and a failing HyDE, must not raise — falls back to base query
    ans = await rag.aask("how are backups done?", hyde=True)
    assert "NAS" in ans.text, "answer must come from base retrieval even if hyde fails"


async def test_ask_from_running_loop_no_runtimeerror(tmp_path, monkeypatch):
    """ask() and chat() called from inside a running event loop must not raise RuntimeError.

    Bare asyncio.run() raises 'This event loop is already running.' in async contexts
    (pytest-asyncio, Jupyter). _run_async() handles this by delegating to a thread."""
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str(notes / "*.md")]),
        index=IndexConfig(path=str(tmp_path / "idx")),
    )

    async def _fake_query(prompt, options):
        yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    from raggity.core import Raggity
    from raggity.conversation import Conversation
    rag = Raggity(cfg)
    rag.ingest()

    # These would raise RuntimeError("This event loop is already running.") with bare asyncio.run
    ans = rag.ask("how are backups done?")
    assert "NAS" in ans.text

    conv = Conversation()
    ans2 = rag.chat(conv, "how are backups done?")
    assert "NAS" in ans2.text


async def test_ask_decompose_from_running_loop(tmp_path, monkeypatch):
    """ask_decompose() from inside a running loop must not raise RuntimeError."""
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str(notes / "*.md")]),
        index=IndexConfig(path=str(tmp_path / "idx")),
    )

    async def _fake(prompt, options):
        if "sub-questions" in prompt or ("Question:" in prompt and "Give at most" in prompt):
            yield _AssistantMessage("how often?\nwhere stored?")
        else:
            yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")

    monkeypatch.setattr(llm_mod, "query", _fake)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    from raggity.core import Raggity
    rag = Raggity(cfg)
    rag.ingest()
    # Would raise RuntimeError with bare asyncio.run
    ans = rag.ask_decompose("how are backups done?")
    assert "NAS" in ans.text


def test_core_chat_two_turns_appends_conversation(tmp_path, monkeypatch):
    """2-turn chat: conversation accumulates 4 turns (user+assistant×2)."""
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str(notes / "*.md")]),
        index=IndexConfig(path=str(tmp_path / "idx")),
    )

    async def _fake_query(prompt, options):
        yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    from raggity.core import Raggity
    from raggity.conversation import Conversation

    rag = Raggity(cfg)
    rag.ingest()
    conv = Conversation()

    ans1 = rag.chat(conv, "how are backups done?")
    assert "NAS" in ans1.text
    assert len(conv.turns) == 2

    ans2 = rag.chat(conv, "where exactly?")
    assert "NAS" in ans2.text
    assert len(conv.turns) == 4
    assert conv.turns[0] == ("user", "how are backups done?")
    assert conv.turns[2] == ("user", "where exactly?")


# ---------------------------------------------------------------------------
# v0.9.0 Task 5a: achat_stream — token deltas + turn recording + disconnect
# ---------------------------------------------------------------------------

def _stream_cfg(tmp_path):
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    return RaggityConfig(sources=SourcesConfig(include=[str(notes / "*.md")]),
                         index=IndexConfig(path=str(tmp_path / "idx")))


def test_achat_stream_yields_deltas_and_records_turns(tmp_path, monkeypatch):
    import asyncio
    from raggity.core import Raggity
    from raggity.conversation import Conversation
    from raggity.models import Answer

    async def _fake(prompt, options):
        yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
    monkeypatch.setattr(llm_mod, "query", _fake)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    rag = Raggity(_stream_cfg(tmp_path)); rag.ingest()
    conv = Conversation()

    async def _run():
        deltas, final = [], None
        async for piece in rag.achat_stream(conv, "how are backups done?"):
            if isinstance(piece, Answer):
                final = piece
            else:
                deltas.append(piece)
        return deltas, final

    deltas, final = asyncio.run(_run())
    assert "".join(deltas)  # streamed some text
    assert final is not None and "NAS" in final.text
    # Turns recorded ONLY after the final Answer.
    assert conv.turns == [("user", "how are backups done?"),
                          ("assistant", final.text)]


def test_achat_stream_disconnect_records_no_turns(tmp_path, monkeypatch):
    import asyncio
    from raggity.core import Raggity
    from raggity.conversation import Conversation

    async def _fake(prompt, options):
        yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
    monkeypatch.setattr(llm_mod, "query", _fake)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    rag = Raggity(_stream_cfg(tmp_path)); rag.ingest()
    conv = Conversation()

    async def _run():
        agen = rag.achat_stream(conv, "how are backups done?")
        await agen.__anext__()  # pull first delta, then simulate client disconnect
        with pytest.raises(asyncio.CancelledError):
            await agen.athrow(asyncio.CancelledError)

    asyncio.run(_run())
    assert conv.turns == []  # no half-turn recorded on mid-stream disconnect


def test_transform_cache_hit_skips_second_expand_call(tmp_path, monkeypatch):
    """With expand + cache on, a repeat question reuses the cached query expansion."""
    from raggity.config import RetrievalConfig
    from raggity.core import Raggity

    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(sources=SourcesConfig(include=[str(notes / "*.md")]),
                        index=IndexConfig(path=str(tmp_path / "idx")),
                        retrieval=RetrievalConfig(expand=True))
    cfg.generation.cache = True

    expand_calls = {"n": 0}

    async def _fake(prompt, options):
        if "alternative phrasings" in prompt:
            expand_calls["n"] += 1
            yield _AssistantMessage("rephrase one\nrephrase two")
        else:
            yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
    monkeypatch.setattr(llm_mod, "query", _fake)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    rag = Raggity(cfg); rag.ingest()
    rag.ask("how are backups done?")
    rag.ask("how are backups done?")  # same question → transform-cache hit
    assert expand_calls["n"] == 1


# ---------------------------------------------------------------------------
# Persona: effective system prompt wiring + cache-key invalidation (v0.10.0)
# ---------------------------------------------------------------------------

def test_core_system_prompt_default_and_persona():
    from raggity.core import Raggity
    from raggity.config import RaggityConfig, GenerationConfig
    from raggity.prompts import SYSTEM_PROMPT
    assert Raggity(RaggityConfig()).system_prompt == SYSTEM_PROMPT
    rag = Raggity(RaggityConfig(generation=GenerationConfig(persona="I am Alex.")))
    assert "I am Alex." in rag.system_prompt
    # The answerer built by the Raggity must carry the effective prompt.
    assert rag.answerer.system_prompt == rag.system_prompt


def test_cache_key_changes_when_persona_toggles():
    """Answer-cache key must include the EFFECTIVE (persona-included) prompt."""
    from raggity.cache import cache_key
    from raggity.core import Raggity
    from raggity.config import RaggityConfig, GenerationConfig
    plain = Raggity(RaggityConfig())
    persona = Raggity(RaggityConfig(generation=GenerationConfig(persona="I am Alex.")))
    ids = ["c1", "c2"]
    k_plain = cache_key("q", ids, "m", system_prompt=plain.system_prompt)
    k_persona = cache_key("q", ids, "m", system_prompt=persona.system_prompt)
    assert k_plain != k_persona
