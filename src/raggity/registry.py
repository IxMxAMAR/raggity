from __future__ import annotations

import importlib

_REGISTRY: dict[tuple[str, str], str] = {}


class BackendNotFound(Exception):
    pass


def register(role: str, name: str, dotted: str) -> None:
    _REGISTRY[(role, name)] = dotted


def import_dotted(dotted: str) -> type:
    """Import ``"package.module:ClassName"`` and return the class.

    Used for user-supplied pluggable components (e.g. a custom reranker named
    directly by dotted path in config) that are not pre-registered by name.
    """
    module_path, _, attr = dotted.partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def resolve(role: str, name: str) -> type:
    dotted = _REGISTRY.get((role, name))
    if dotted is None:
        raise BackendNotFound(
            f"No {role} backend named {name!r} is registered."
        )
    module_path, _, attr = dotted.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise BackendNotFound(
            f"The {role} backend {name!r} needs an optional dependency. "
            f"Try: pip install raggity[{name}]"
        ) from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise BackendNotFound(
            f"The {role} backend {name!r} registered class {attr!r} was not found "
            f"in module {module_path!r}. The package may need updating: "
            f"pip install -U raggity[{name}]"
        ) from exc
