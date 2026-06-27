from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .chunker import chunk_document
from .loader import load_documents


@dataclass
class IngestReport:
    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0


class Indexer:
    def __init__(self, embedder, store, manifest_path: str) -> None:
        self.embedder = embedder
        self.store = store
        self.manifest_path = manifest_path

    def _load_manifest(self) -> dict[str, str]:
        if os.path.isfile(self.manifest_path):
            with open(self.manifest_path, encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _save_manifest(self, manifest: dict[str, str]) -> None:
        Path(self.manifest_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)

    def ingest(self, globs: list[str]) -> IngestReport:
        report = IngestReport()
        manifest = self._load_manifest()
        docs = load_documents(globs)
        seen: dict[str, str] = {}

        for doc in docs:
            seen[doc.path] = doc.file_hash
            prev = manifest.get(doc.path)
            if prev == doc.file_hash:
                report.unchanged += 1
                continue
            # changed or new → replace chunks for this source
            self.store.delete_source(doc.path)
            self.store.upsert(chunk_document(doc), self.embedder)
            if prev is None:
                report.added += 1
            else:
                report.updated += 1

        # delete sources that vanished from disk
        for old_path in list(manifest.keys()):
            if old_path not in seen:
                self.store.delete_source(old_path)
                report.deleted += 1

        self._save_manifest(seen)
        self.store.optimize()
        return report
