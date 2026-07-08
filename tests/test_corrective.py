"""CRAG-style corrective retrieval loop (retrieval.corrective, opt-in).

The mock chokepoint is ``raggity.llm.query``. Prompts are dispatched by marker:
  - ``"Retrieved passages"`` in prompt  -> retrieval evaluator (CRAG verdict)
  - prompt startswith ``"Query:"``       -> query rewrite
  - otherwise                            -> final answer (build_user_prompt)
"""
import asyncio

import raggity.llm as llm_mod
from raggity.config import (RaggityConfig, SourcesConfig, IndexConfig,
                            RetrievalConfig)
from raggity.core import Raggity
from raggity.models import Answer, Chunk
from raggity.prompts import ABSTAIN_MESSAGE


class _Block:
    def __init__(self, text): self.text = text


class _AM:
    def __init__(self, text): self.content = [_Block(text)]


def _mock(monkeypatch, verdict="correct", rewritten="rewritten NAS query",
          answer="Backups run nightly to the NAS."):
    """Install the llm chokepoint mock; return a dict counting each call type."""
    seen = {"evaluate": 0, "rewrite": 0, "answer": 0}

    async def _fake(prompt, options):
        if "Retrieved passages" in prompt:
            seen["evaluate"] += 1
            yield _AM(verdict)
        elif prompt.startswith("Query:"):
            seen["rewrite"] += 1
            yield _AM(rewritten)
        else:
            seen["answer"] += 1
            yield _AM(answer)

    monkeypatch.setattr(llm_mod, "query", _fake)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AM)
    return seen


def _c(cid, score):
    return Chunk("text about " + cid, "a.md", "A", "A", 0, cid, score=score)


def _cfg(tmp_path, corrective=True, rerank=False, cache=False):
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[]),
        index=IndexConfig(path=str(tmp_path / "idx")),
        retrieval=RetrievalConfig(corrective=corrective, rerank=rerank),
    )
    cfg.generation.cache = cache
    return cfg


def _spy_retrieval(monkeypatch, rag, first, second):
    """Patch retriever.retrieve_multi: 1-query call -> *first*, 2-query -> *second*.
    Records the queries lists passed on each call."""
    calls = []

    def _spy(queries, rerank_query, **kw):
        calls.append(list(queries))
        return list(first) if len(queries) == 1 else list(second)

    monkeypatch.setattr(rag.retriever, "retrieve_multi", _spy)
    return calls


def _spy_answer(monkeypatch, rag):
    captured = []
    orig = rag.answerer.answer

    async def _spy(question, chunks, **kw):
        captured.append(list(chunks))
        return await orig(question, chunks, **kw)

    monkeypatch.setattr(rag.answerer, "answer", _spy)
    return captured


# --- 1. default off: no evaluator, behavior unchanged ----------------------

def test_corrective_off_by_default_no_evaluator(tmp_path, monkeypatch):
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(sources=SourcesConfig(include=[str(notes / "*.md")]),
                        index=IndexConfig(path=str(tmp_path / "idx")))
    assert cfg.retrieval.corrective is False
    seen = _mock(monkeypatch)
    rag = Raggity(cfg); rag.ingest()
    ans = rag.ask("how are backups done?")
    assert "NAS" in ans.text
    assert seen["evaluate"] == 0  # evaluator never fires when corrective off


# --- 2. verdict "correct": no second retrieval, no rewrite -----------------

def test_verdict_correct_skips_corrective_round(tmp_path, monkeypatch):
    seen = _mock(monkeypatch, verdict="correct")
    rag = Raggity(_cfg(tmp_path))
    calls = _spy_retrieval(monkeypatch, rag, [_c("c_a", 0.6), _c("c_b", 0.5)], [])
    captured = _spy_answer(monkeypatch, rag)
    ans = rag.ask("how are backups done?")
    assert "NAS" in ans.text
    assert seen["evaluate"] == 1
    assert seen["rewrite"] == 0
    assert calls == [["how are backups done?"]]  # only the first-round retrieval
    assert {c.chunk_id for c in captured[0]} == {"c_a", "c_b"}


# --- 3. verdict "incorrect": one rewrite + merge/dedup/reslice --------------

def test_verdict_incorrect_triggers_rewrite_and_merge(tmp_path, monkeypatch):
    seen = _mock(monkeypatch, verdict="incorrect", rewritten="NAS backup schedule")
    rag = Raggity(_cfg(tmp_path))
    first = [_c("c_a", 0.5), _c("c_b", 0.4)]
    second = [_c("c_b", 0.6), _c("c_c", 0.7)]  # c_b overlaps -> dedup
    calls = _spy_retrieval(monkeypatch, rag, first, second)
    captured = _spy_answer(monkeypatch, rag)
    ans = rag.ask("how are backups done?")
    assert "NAS" in ans.text
    assert seen["evaluate"] == 1
    assert seen["rewrite"] == 1
    # Second retrieval keeps the ORIGINAL query in the fusion + the rewrite.
    assert calls == [["how are backups done?"],
                     ["how are backups done?", "NAS backup schedule"]]
    # Answer sees the MERGED, deduped pool (c_b appears once).
    assert {c.chunk_id for c in captured[0]} == {"c_a", "c_b", "c_c"}


def test_verdict_ambiguous_also_triggers_corrective(tmp_path, monkeypatch):
    seen = _mock(monkeypatch, verdict="ambiguous")
    rag = Raggity(_cfg(tmp_path))
    calls = _spy_retrieval(monkeypatch, rag, [_c("c_a", 0.5)], [_c("c_d", 0.6)])
    rag.ask("how are backups done?")
    assert seen["rewrite"] == 1
    assert len(calls) == 2


# --- 4. empty first retrieval: one corrective shot, still empty -> abstain --

def test_empty_first_retrieval_gets_one_shot_then_abstains(tmp_path, monkeypatch):
    seen = _mock(monkeypatch, verdict="incorrect")
    rag = Raggity(_cfg(tmp_path))
    calls = _spy_retrieval(monkeypatch, rag, [], [])  # both rounds empty
    ans = rag.ask("how are backups done?")
    assert seen["evaluate"] == 0  # empty set -> evaluator skipped
    assert seen["rewrite"] == 1   # corrective still gets its rewrite shot
    assert len(calls) == 2        # rewritten-query retrieval attempted
    assert ans.abstained is True
    assert ans.text == ABSTAIN_MESSAGE


# --- 5. evaluator failure: fall back to original chunks, warn, no crash -----

def test_evaluator_failure_falls_back_to_original(tmp_path, monkeypatch, caplog):
    seen = _mock(monkeypatch, verdict="incorrect")
    rag = Raggity(_cfg(tmp_path))
    first = [_c("c_a", 0.6)]
    calls = _spy_retrieval(monkeypatch, rag, first, [_c("c_z", 0.9)])
    captured = _spy_answer(monkeypatch, rag)

    import raggity.query_transform as qt

    async def _boom(question, chunks, provider):
        raise RuntimeError("evaluator LLM down")

    monkeypatch.setattr(qt, "evaluate_retrieval", _boom)

    with caplog.at_level("WARNING", logger="raggity.core"):
        ans = rag.ask("how are backups done?")
    assert "NAS" in ans.text
    assert seen["rewrite"] == 0            # no corrective round on failure
    assert calls == [["how are backups done?"]]
    assert {c.chunk_id for c in captured[0]} == {"c_a"}  # original chunks used
    assert any("corrective evaluator failed" in r.message for r in caplog.records)


# --- 6. verdict caching: same question+chunks -> one evaluator call ---------

def test_verdict_cached_across_repeat_question(tmp_path, monkeypatch):
    seen = _mock(monkeypatch, verdict="correct")
    rag = Raggity(_cfg(tmp_path, cache=True))
    stable = [_c("c_a", 0.6), _c("c_b", 0.5)]
    _spy_retrieval(monkeypatch, rag, stable, [])
    rag.ask("how are backups done?")
    rag.ask("how are backups done?")  # same q + chunks -> verdict cache hit
    assert seen["evaluate"] == 1


# --- 7. aask_stream parity: corrective works while streaming ----------------

def test_aask_stream_corrective_parity(tmp_path, monkeypatch):
    seen = _mock(monkeypatch, verdict="incorrect", rewritten="NAS backup schedule")
    rag = Raggity(_cfg(tmp_path))
    first = [_c("c_a", 0.5)]
    second = [_c("c_c", 0.7)]
    calls = _spy_retrieval(monkeypatch, rag, first, second)

    async def _run():
        deltas, final = [], None
        async for piece in rag.aask_stream("how are backups done?"):
            if isinstance(piece, Answer):
                final = piece
            else:
                deltas.append(piece)
        return deltas, final

    deltas, final = asyncio.run(_run())
    assert "".join(deltas)
    assert final is not None and "NAS" in final.text
    assert seen["evaluate"] == 1
    assert seen["rewrite"] == 1
    assert calls == [["how are backups done?"],
                     ["how are backups done?", "NAS backup schedule"]]


# --- 8. rewrite memoisation: repeat corrective-triggered question -> one rewrite

def test_rewrite_cached_across_repeat_question(tmp_path, monkeypatch):
    """With cache on, a repeated corrective-triggered question must not re-pay
    the rewrite call — deterministic rewrites keep the answer-cache key stable."""
    seen = _mock(monkeypatch, verdict="incorrect", rewritten="NAS backup schedule")
    rag = Raggity(_cfg(tmp_path, cache=True))
    first = [_c("c_a", 0.5)]
    second = [_c("c_c", 0.7)]
    _spy_retrieval(monkeypatch, rag, first, second)
    rag.ask("how are backups done?")
    rag.ask("how are backups done?")
    assert seen["rewrite"] == 1
