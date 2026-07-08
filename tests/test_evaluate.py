import pytest
from raggity.models import Chunk
from raggity.evaluate import evaluate, EvalResult, load_golden


class FakeRetriever:
    def retrieve(self, q):
        if "backup" in q:
            return [Chunk(text="t", source_path="a.md", title="A",
                          heading_path="A", ordinal=0, chunk_id="c1")]
        return []  # abstain for the negative


def test_load_golden(fixtures_dir):
    rows = load_golden(str(fixtures_dir / "golden.jsonl"))
    assert rows[0]["question"].startswith("how are backups")


def test_evaluate_hit_and_mrr():
    golden = [{"question": "how are backups done?", "relevant_source_paths": ["a.md"]}]
    res = evaluate(FakeRetriever(), golden, k=5)
    assert isinstance(res, EvalResult)
    assert res.hit_rate == 1.0 and res.mrr == 1.0 and res.n == 1


def test_evaluate_negative_correctly_zero_when_expected_empty():
    # a question whose gold is empty: retrieving nothing is a "hit" (correct abstain)
    golden = [{"question": "nope", "relevant_source_paths": []}]
    res = evaluate(FakeRetriever(), golden, k=5)
    assert res.hit_rate == 1.0


def test_evaluate_recall_positive():
    from raggity.evaluate import evaluate
    golden = [{"question": "how are backups done?", "relevant_source_paths": ["a.md"]}]
    res = evaluate(FakeRetriever(), golden, k=5)
    assert res.recall == 1.0


def test_evaluate_n_zero():
    from raggity.evaluate import evaluate, EvalResult
    res = evaluate(FakeRetriever(), [], k=5)
    assert res == EvalResult(0.0, 0.0, 0.0, 0)


def test_llm_judge_averages_verdicts(tmp_path, monkeypatch):
    import asyncio, raggity.llm as llm_mod
    from raggity.evaluate import llm_judge, JudgeResult
    from raggity.config import RaggityConfig, SourcesConfig, IndexConfig
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(sources=SourcesConfig(include=[str(notes / "*.md")]),
                        index=IndexConfig(path=str(tmp_path / "idx")))

    class _Block:
        def __init__(self, t): self.text = t
    class _AM:
        def __init__(self, t): self.content = [_Block(t)]

    # Both answerer and judge go through llm_mod.query (via ClaudeProvider).
    # The merged judge expects two labeled lines; the answerer just echoes them.
    async def _fake(prompt, options):
        yield _AM("FAITHFUL: YES\nRELEVANT: YES")
    monkeypatch.setattr(llm_mod, "query", _fake)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AM)

    from raggity.core import Raggity
    rag = Raggity(cfg); rag.ingest()
    golden = [{"question": "how are backups done?", "relevant_source_paths": ["a.md"]}]
    res = asyncio.run(llm_judge(rag, golden, rag.provider))
    assert isinstance(res, JudgeResult)
    assert res.faithfulness == 1.0 and res.answer_relevance == 1.0 and res.n == 1


# ---------------------------------------------------------------------------
# Fix 4: golden-row schema validation + empty-golden div-by-zero guard
# ---------------------------------------------------------------------------

def test_parse_judge_truth_table():
    """_parse_judge reads two labeled lines; missing/garbled → False (NO)."""
    from raggity.evaluate import _parse_judge
    assert _parse_judge("FAITHFUL: YES\nRELEVANT: YES") == (True, True)
    assert _parse_judge("FAITHFUL: NO\nRELEVANT: YES") == (False, True)
    assert _parse_judge("FAITHFUL: YES\nRELEVANT: NO") == (True, False)
    # case-insensitive + surrounding prose lines ignored
    assert _parse_judge("here you go\nfaithful: yes\nrelevant: no") == (True, False)
    # garbled / missing lines default to False
    assert _parse_judge("totally unparseable") == (False, False)
    assert _parse_judge("FAITHFUL: YES") == (True, False)
    assert _parse_judge("") == (False, False)


def test_evaluate_malformed_row_raises_with_row_index():
    """A row missing 'question' raises ValueError mentioning the row index."""
    golden = [
        {"question": "ok?", "relevant_source_paths": []},
        {"TYPO_question": "bad row"},  # row index 1
    ]
    with pytest.raises((ValueError, KeyError)) as exc:
        evaluate(FakeRetriever(), golden, k=5)
    assert "1" in str(exc.value) or "row" in str(exc.value).lower()


def test_evaluate_empty_golden_no_div_by_zero():
    """evaluate() on an empty golden set returns zeros, no ZeroDivisionError."""
    res = evaluate(FakeRetriever(), [], k=5)
    assert res == EvalResult(0.0, 0.0, 0.0, 0)


def test_llm_judge_empty_golden_no_div_by_zero():
    """llm_judge() on an empty golden set returns zeros, no ZeroDivisionError."""
    import asyncio
    from raggity.evaluate import llm_judge, JudgeResult

    class FakeProv:
        async def complete(self, sys, prompt): return "YES"

    class FakeRag:
        class retriever:
            @staticmethod
            def retrieve(q): return []
        class answerer:
            @staticmethod
            async def answer(q, chunks):
                class R:
                    text = ""; abstained = True; citations = []
                return R()

    res = asyncio.run(llm_judge(FakeRag(), [], FakeProv()))
    assert isinstance(res, JudgeResult)
    assert res.n == 0


# ---------------------------------------------------------------------------
# Task 6a: hallucination-resistance metrics via unanswerable golden rows
# ---------------------------------------------------------------------------

def test_validate_row_accepts_answerable_flag():
    """answerable=True/False are both accepted; a non-bool value raises."""
    from raggity.evaluate import _validate_row
    _validate_row({"question": "q", "answerable": True}, 0)
    _validate_row({"question": "q", "answerable": False}, 0)
    _validate_row({"question": "q"}, 0)  # absent = fine (defaults True downstream)
    with pytest.raises(ValueError):
        _validate_row({"question": "q", "answerable": "nope"}, 0)


def test_evaluate_skips_unanswerable_rows_from_retrieval_metrics():
    """Unanswerable rows are excluded from Hit@k/MRR/Recall@k and counted separately."""
    golden = [
        {"question": "how are backups done?", "relevant_source_paths": ["a.md"]},
        {"question": "what colour is the CEO's car?", "answerable": False},
    ]
    res = evaluate(FakeRetriever(), golden, k=5)
    assert res.n == 1  # only the answerable row counts toward the denominator
    assert res.hit_rate == 1.0 and res.mrr == 1.0 and res.recall == 1.0
    assert res.unanswerable_total == 1


def test_evaluate_no_unanswerable_rows_defaults_to_zero():
    """Golden sets with no 'answerable' key at all: unanswerable_total is 0."""
    golden = [{"question": "how are backups done?", "relevant_source_paths": ["a.md"]}]
    res = evaluate(FakeRetriever(), golden, k=5)
    assert res.unanswerable_total == 0


def test_evaluate_all_unanswerable_no_div_by_zero():
    golden = [{"question": "nope", "answerable": False}]
    res = evaluate(FakeRetriever(), golden, k=5)
    assert res.n == 0
    assert res.unanswerable_total == 1
    assert res.hit_rate == 0.0 and res.mrr == 0.0 and res.recall == 0.0


class _JudgeAnswer:
    def __init__(self, text="", abstained=False):
        self.text = text
        self.abstained = abstained
        self.citations = []


def test_llm_judge_rejection_and_false_answer_rate():
    """One correct abstention + one hallucinated answer on unanswerable rows."""
    import asyncio
    from raggity.evaluate import llm_judge, JudgeResult

    golden = [
        {"question": "q-abstain", "answerable": False},
        {"question": "q-answer", "answerable": False},
    ]

    class FakeRetrieverAlwaysSome:
        @staticmethod
        def retrieve(q):
            return [Chunk(text="t", source_path="a.md", title="A",
                          heading_path="A", ordinal=0, chunk_id="c1")]

    class FakeAnswerer:
        @staticmethod
        async def answer(q, chunks):
            if q == "q-abstain":
                return _JudgeAnswer(text="I don't have enough information.", abstained=True)
            return _JudgeAnswer(text="a confidently made-up answer", abstained=False)

    class FakeRag:
        retriever = FakeRetrieverAlwaysSome()
        answerer = FakeAnswerer()

    class FakeProv:
        async def complete(self, sys, prompt):
            raise AssertionError("provider.complete must not be called for unanswerable rows")

    res = asyncio.run(llm_judge(FakeRag(), golden, FakeProv()))
    assert isinstance(res, JudgeResult)
    assert res.n == 0  # no answerable rows in this golden set
    assert res.rejection_rate == 0.5
    assert res.false_answer_rate == 0.5


def test_llm_judge_no_unanswerable_rows_new_fields_none_and_old_numbers_unchanged(
        tmp_path, monkeypatch):
    """A purely-answerable golden set: rejection/false-answer fields are None,
    and faithfulness/answer_relevance/n are computed exactly as before this feature."""
    import asyncio, raggity.llm as llm_mod
    from raggity.evaluate import llm_judge
    from raggity.config import RaggityConfig, SourcesConfig, IndexConfig

    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = RaggityConfig(sources=SourcesConfig(include=[str(notes / "*.md")]),
                        index=IndexConfig(path=str(tmp_path / "idx")))

    class _Block:
        def __init__(self, t): self.text = t
    class _AM:
        def __init__(self, t): self.content = [_Block(t)]

    async def _fake(prompt, options):
        yield _AM("FAITHFUL: YES\nRELEVANT: YES")
    monkeypatch.setattr(llm_mod, "query", _fake)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AM)

    from raggity.core import Raggity
    rag = Raggity(cfg); rag.ingest()
    golden = [{"question": "how are backups done?", "relevant_source_paths": ["a.md"]}]
    res = asyncio.run(llm_judge(rag, golden, rag.provider))
    assert res.faithfulness == 1.0 and res.answer_relevance == 1.0 and res.n == 1
    assert res.rejection_rate is None
    assert res.false_answer_rate is None
