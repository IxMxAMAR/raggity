from dataclasses import dataclass, field


@dataclass
class Document:
    path: str
    title: str
    text: str
    file_hash: str
    mtime: float
    size: int = 0


@dataclass
class Chunk:
    text: str
    source_path: str
    title: str
    heading_path: str
    ordinal: int
    chunk_id: str
    score: float = 0.0
    parent_id: str = ""
    parent_text: str = ""
    vector: list[float] | None = field(default=None, repr=False, compare=False)


@dataclass
class Citation:
    chunk_id: str
    source_path: str
    title: str
    supported: bool


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    abstained: bool = False
