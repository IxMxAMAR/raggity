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
