import pytest
from raggity.registry import register, resolve, BackendNotFound


def test_resolve_builtin_class():
    register("embedder", "dummy", "raggity.models:Chunk")
    cls = resolve("embedder", "dummy")
    assert cls.__name__ == "Chunk"


def test_unknown_name_raises_with_hint():
    with pytest.raises(BackendNotFound) as exc:
        resolve("embedder", "nope")
    assert "nope" in str(exc.value)


def test_missing_module_raises_with_pip_hint():
    register("store", "ghost", "raggity_ghost_pkg:Thing")
    with pytest.raises(BackendNotFound) as exc:
        resolve("store", "ghost")
    assert "pip install" in str(exc.value)
