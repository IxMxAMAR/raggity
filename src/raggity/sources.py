"""Add a folder to the config without flattening the file it lives in.

The config is something a user edits by hand, so it is round-tripped through
tomlkit: comments, ordering and unknown keys all survive an append.
"""
from __future__ import annotations

from pathlib import Path

import tomlkit


def glob_for(path: Path) -> str:
    """The recursive pattern that indexes everything under a folder."""
    return Path(path).as_posix().rstrip("/") + "/**/*"


def add_sources(paths: list[Path], config: str | None = None) -> Path:
    """Append *paths* to `[sources] include`, creating the config if needed."""
    from .cli import _INIT_TEMPLATE
    from .config import _find_config_path

    dest = Path(config) if config else _find_config_path(None)
    if dest is None:
        dest = Path.cwd() / "raggity.toml"
    created = not dest.exists()
    if created:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_INIT_TEMPLATE, encoding="utf-8")

    doc = tomlkit.parse(dest.read_text(encoding="utf-8"))
    sources = doc.get("sources")
    if sources is None:
        sources = tomlkit.table()
        doc["sources"] = sources
    include = sources.get("include")
    if include is None or created:
        # The template seeds `include` with RELATIVE globs (`**/*.md`), which
        # resolve against whatever directory the command ran from. Someone who
        # said `rag add ~/notes` asked for one folder; inheriting those would
        # quietly index the working directory alongside it. Keep the template
        # for its comments and structure, not for its example patterns.
        include = tomlkit.array()
        sources["include"] = include

    existing = {str(x) for x in include}
    for p in paths:
        pattern = glob_for(p)
        if pattern not in existing:
            include.append(pattern)
            existing.add(pattern)

    dest.write_text(tomlkit.dumps(doc), encoding="utf-8")
    return dest
