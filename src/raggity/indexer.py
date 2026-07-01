from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import chunk_document
from .loader import load_documents


@dataclass
class IngestReport:
    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    skipped_needs_extra: dict[str, int] = field(default_factory=dict)
    skipped_generic: int = 0


class Indexer:
    def __init__(self, embedder, store, manifest_path: str, fingerprint: str = "",
                 chunk_kwargs: dict | None = None, ann_threshold: int = 0) -> None:
        self.embedder = embedder
        self.store = store
        self.manifest_path = manifest_path
        self.fingerprint = fingerprint
        self.chunk_kwargs = chunk_kwargs
        self.ann_threshold = ann_threshold

    def _load_manifest(self) -> dict[str, str]:
        if os.path.isfile(self.manifest_path):
            with open(self.manifest_path, encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _save_manifest(self, manifest: dict[str, str]) -> None:
        Path(self.manifest_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)

    def _fp_path(self) -> str:
        return self.manifest_path + ".fingerprint"

    def _fingerprint_changed(self) -> bool:
        if not self.fingerprint:
            return False
        prev = None
        if os.path.isfile(self._fp_path()):
            with open(self._fp_path(), encoding="utf-8") as fh:
                prev = fh.read().strip()
        if prev != self.fingerprint:
            return True
        return False

    def _write_fingerprint(self) -> None:
        if self.fingerprint:
            Path(self._fp_path()).parent.mkdir(parents=True, exist_ok=True)
            with open(self._fp_path(), "w", encoding="utf-8") as fh:
                fh.write(self.fingerprint)

    def _clear_manifest(self) -> None:
        if os.path.isfile(self.manifest_path):
            os.remove(self.manifest_path)

    def ingest(self, globs: list[str]) -> IngestReport:
        report = IngestReport()
        if self._fingerprint_changed():
            self.store.reset()
            self._clear_manifest()
        manifest = self._load_manifest()
        docs, skipped_needs_extra, skipped_generic = load_documents(globs)
        # Accumulate skip counts into the report
        for extra, cnt in skipped_needs_extra.items():
            report.skipped_needs_extra[extra] = (
                report.skipped_needs_extra.get(extra, 0) + cnt
            )
        report.skipped_generic += skipped_generic
        seen: dict[str, str] = {}

        all_chunks: list = []
        for doc in docs:
            seen[doc.path] = doc.file_hash
            prev = manifest.get(doc.path)
            if prev == doc.file_hash:
                report.unchanged += 1
                continue
            # changed or new → delete existing chunks for this source
            self.store.delete_source(doc.path)
            all_chunks.extend(chunk_document(doc, **(self.chunk_kwargs or {})))
            if prev is None:
                report.added += 1
            else:
                report.updated += 1

        # delete sources that vanished from disk
        for old_path in list(manifest.keys()):
            if old_path not in seen:
                self.store.delete_source(old_path)
                report.deleted += 1

        # single batched upsert across all changed files
        if all_chunks:
            self.store.upsert(all_chunks, self.embedder)

        self.store.ensure_ann_index(self.ann_threshold)
        self._save_manifest(seen)
        self._write_fingerprint()
        self.store.optimize()
        return report
