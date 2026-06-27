from raggity.models import Answer, Citation
from raggity import cache


def test_cache_key_stable_and_order_independent():
    k1 = cache.cache_key("q", ["b", "a"], "m")
    k2 = cache.cache_key("q", ["a", "b"], "m")
    assert k1 == k2  # chunk id order independent
    assert cache.cache_key("q", ["a"], "m") != cache.cache_key("q2", ["a"], "m")


def test_answer_roundtrip():
    a = Answer(text="x", citations=[Citation("c1", "a.md", "A", True)], abstained=False)
    d = cache.answer_to_dict(a)
    a2 = cache.answer_from_dict(d)
    assert a2.text == "x" and a2.citations[0].chunk_id == "c1" and a2.abstained is False


def test_load_tolerates_missing_and_corrupt(tmp_path):
    p = str(tmp_path / "answer_cache.json")
    assert cache.load(p) == {}
    with open(p, "w") as fh:
        fh.write("{ not json")
    assert cache.load(p) == {}
