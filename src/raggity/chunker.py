from __future__ import annotations

import hashlib
import re

from .models import Chunk, Document

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def estimate_tokens(text: str) -> int:
    # Dependency-free heuristic; tiktoken is wrong for Claude and unneeded here.
    return max(1, len(text) // 4)


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Return (heading_path, section_body) preserving markdown header hierarchy."""
    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(stack), body))

    for line in text.splitlines():
        m = _HEADER_RE.match(line)
        if m:
            flush()
            buf = []
            level = len(m.group(1))
            title = m.group(2).strip()
            stack = stack[: level - 1]
            while len(stack) < level - 1:
                stack.append("")
            stack.append(title)
        else:
            buf.append(line)
    flush()
    if not sections:
        sections.append(("", text.strip()))
    return sections


def _hard_split_paragraph(text: str, max_chars: int) -> list[str]:
    """Slice an oversize paragraph into pieces of <= *max_chars* chars.

    Prefer breaking at whitespace near the boundary (rfind within the last 200
    chars) so words are not cut mid-way; fall back to a hard character cut when
    no whitespace is present.  Guarantees every returned slice is <= max_chars.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + max_chars, n)
        if end < n:
            ws = text.rfind(" ", max(i, end - 200), end)
            if ws > i:
                end = ws
        piece = text[i:end].strip()
        if piece:
            out.append(piece)
        i = end
        while i < n and text[i] == " ":
            i += 1
    return out


def _split_body(body: str, target_tokens: int, overlap_tokens: int) -> list[str]:
    raw = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    # Hard-cap: a single paragraph larger than target_tokens would otherwise be
    # emitted as one colossal chunk (which crashed onnxruntime attention on a
    # giant single-paragraph .txt).  Pre-split any oversize paragraph so NO
    # chunk can ever exceed ~target_tokens, regardless of input shape.  Only
    # activates for oversize paragraphs; normal inputs are untouched.
    max_chars = target_tokens * 4  # estimate_tokens == len // 4
    paras: list[str] = []
    for p in raw:
        if estimate_tokens(p) > target_tokens:
            paras.extend(_hard_split_paragraph(p, max_chars))
        else:
            paras.append(p)
    pieces: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for para in paras:
        ptok = estimate_tokens(para)
        if cur and cur_tok + ptok > target_tokens:
            pieces.append("\n\n".join(cur))
            # overlap: carry tail paragraphs until ~overlap_tokens
            carry: list[str] = []
            ctok = 0
            for prev in reversed(cur):
                carry.insert(0, prev)
                ctok += estimate_tokens(prev)
                if ctok >= overlap_tokens:
                    break
            cur = carry[:]
            cur_tok = sum(estimate_tokens(x) for x in cur)
        cur.append(para)
        cur_tok += ptok
    if cur:
        pieces.append("\n\n".join(cur))
    return pieces or [body]


def _chunk_flat(doc: Document, target_tokens: int, overlap_tokens: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordinal = 0
    for heading_path, body in _split_into_sections(doc.text):
        full_path = doc.title if not heading_path else f"{doc.title} > {heading_path}"
        for piece in _split_body(body, target_tokens, overlap_tokens):
            header = full_path
            chunk_text = f"{header}\n\n{piece}" if header else piece
            chunk_id = hashlib.sha256(
                f"{doc.path}|{ordinal}|{piece}".encode("utf-8")
            ).hexdigest()
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source_path=doc.path,
                    title=doc.title,
                    heading_path=heading_path or doc.title,
                    ordinal=ordinal,
                    chunk_id=chunk_id,
                )
            )
            ordinal += 1
    return chunks


def _chunk_parent(doc: Document, parent_tokens: int, child_tokens: int,
                  overlap_tokens: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordinal = 0
    parent_index = 0
    for heading_path, body in _split_into_sections(doc.text):
        full_path = doc.title if not heading_path else f"{doc.title} > {heading_path}"
        for parent_piece in _split_body(body, parent_tokens, overlap_tokens):
            parent_text = f"{full_path}\n\n{parent_piece}" if full_path else parent_piece
            parent_id = hashlib.sha256(
                f"{doc.path}|parent|{parent_index}".encode("utf-8")
            ).hexdigest()
            for child_piece in _split_body(parent_piece, child_tokens, overlap_tokens):
                child_text = f"{full_path}\n\n{child_piece}" if full_path else child_piece
                chunk_id = hashlib.sha256(
                    f"{doc.path}|{ordinal}|{child_piece}".encode("utf-8")
                ).hexdigest()
                chunks.append(Chunk(
                    text=child_text, source_path=doc.path, title=doc.title,
                    heading_path=heading_path or doc.title, ordinal=ordinal,
                    chunk_id=chunk_id, parent_id=parent_id, parent_text=parent_text,
                ))
                ordinal += 1
            parent_index += 1
    return chunks


def chunk_document(doc: Document, target_tokens: int = 512,
                   overlap_tokens: int = 64, parent_document: bool = False,
                   parent_target_tokens: int = 1024,
                   child_target_tokens: int = 256) -> list[Chunk]:
    if not parent_document:
        return _chunk_flat(doc, target_tokens, overlap_tokens)
    return _chunk_parent(doc, parent_target_tokens, child_target_tokens, overlap_tokens)
