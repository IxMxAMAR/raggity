from __future__ import annotations

import importlib

_REGISTRY: dict[tuple[str, str], str] = {}


class BackendNotFound(Exception):
    pass


def register(role: str, name: str, dotted: str) -> None:
    _REGISTRY[(role, name)] = dotted


def resolve(role: str, name: str):
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
    return getattr(module, attr)
