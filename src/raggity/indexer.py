from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import chunk_document
from .loader import load_paths, scan_sources


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

    def _load_manifest(self) -> dict[str, dict]:
        """Load the manifest, migrating v1 (flat ``{path: hash}``) to v2.

        v2 shape is ``{path: {"hash", "mtime", "size"}}``.  Migration is a
        value-type sniff: a string value is a v1 hash → wrapped as
        ``{"hash": value}`` (no mtime/size), so it lands in the scan candidates
        and is upgraded in place on first ingest.
        """
        if os.path.isfile(self.manifest_path):
            with open(self.manifest_path, encoding="utf-8") as fh:
                raw = json.load(fh)
            return {
                p: ({"hash": v} if isinstance(v, str) else v)
                for p, v in raw.items()
            }
        return {}

    def _save_manifest(self, manifest: dict[str, dict]) -> None:
        Path(self.manifest_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        os.replace(tmp, self.manifest_path)

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

        # 1) Cheap glob+stat sweep — stat-unchanged files never get hashed/parsed.
        scan = scan_sources(globs, manifest)
        report.skipped_generic += scan.skipped_generic

        new_manifest: dict[str, dict] = {}
        # Carry forward stat-unchanged entries verbatim.
        new_manifest.update(scan.unchanged)
        report.unchanged += len(scan.unchanged)

        # 2) Hash + (conditionally) parse the candidates.
        load = load_paths(scan.candidates, manifest)
        for extra, cnt in load.skipped_needs_extra.items():
            report.skipped_needs_extra[extra] = (
                report.skipped_needs_extra.get(extra, 0) + cnt
            )
        report.skipped_generic += load.skipped_generic

        # Content-unchanged-despite-stat: refresh (mtime, size), keep hash, no parse.
        new_manifest.update(load.unchanged_by_hash)
        report.unchanged += len(load.unchanged_by_hash)

        to_delete: set[str] = set()
        all_chunks: list = []
        for doc in load.docs:
            prev = manifest.get(doc.path)
            # changed → remove old chunks (batched below before upsert).
            to_delete.add(doc.path)
            all_chunks.extend(chunk_document(doc, **(self.chunk_kwargs or {})))
            new_manifest[doc.path] = {
                "hash": doc.file_hash, "mtime": doc.mtime, "size": doc.size,
            }
            if prev is None:
                report.added += 1
            else:
                report.updated += 1

        # 3) Deletion detection via the scan's present-set (glob+stat only).
        for old_path in list(manifest.keys()):
            if old_path not in scan.present:
                to_delete.add(old_path)
                report.deleted += 1

        # single batched delete across all changed + vanished sources
        if to_delete:
            self.store.delete_sources(list(to_delete))

        # single batched upsert across all changed files
        if all_chunks:
            self.store.upsert(all_chunks, self.embedder)

        store_changed = bool(all_chunks) or report.deleted > 0
        if store_changed:
            self.store.ensure_ann_index(self.ann_threshold)
        self._save_manifest(new_manifest)
        self._write_fingerprint()
        if store_changed:
            self.store.optimize()
        return report
