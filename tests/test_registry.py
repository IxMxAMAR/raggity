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


# ---------------------------------------------------------------------------
# Fix 5: missing attribute → BackendNotFound (not AttributeError)
# ---------------------------------------------------------------------------

def test_missing_attr_raises_backend_not_found_not_attribute_error():
    """A valid module but non-existent class → BackendNotFound with install hint."""
    # raggity.models exists; 'NonExistentClass' does not
    register("embedder", "bad_attr", "raggity.models:NonExistentClass")
    with pytest.raises(BackendNotFound) as exc:
        resolve("embedder", "bad_attr")
    msg = str(exc.value)
    assert "NonExistentClass" in msg or "bad_attr" in msg
    # Must NOT be a raw AttributeError
    assert exc.type is BackendNotFound
