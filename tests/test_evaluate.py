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
    import asyncio, raggity.answerer as answerer_mod
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

    # answerer returns an answer; judges return YES
    async def _ans(prompt, options):
        yield _AM("Backups run nightly to the NAS [doc_1#00000000].")
    monkeypatch.setattr(answerer_mod, "query", _ans)
    monkeypatch.setattr(answerer_mod, "AssistantMessage", _AM)
    import raggity.evaluate as ev
    async def _judge(prompt, options):
        yield _AM("YES")
    monkeypatch.setattr(ev, "query", _judge)
    monkeypatch.setattr(ev, "AssistantMessage", _AM)

    from raggity.core import Raggity
    rag = Raggity(cfg); rag.ingest()
    golden = [{"question": "how are backups done?", "relevant_source_paths": ["a.md"]}]
    res = asyncio.run(llm_judge(rag, golden))
    assert isinstance(res, JudgeResult)
    assert res.faithfulness == 1.0 and res.answer_relevance == 1.0 and res.n == 1
