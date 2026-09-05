"""Offer what is on the machine when nothing is indexed yet.

Nothing here indexes without a yes, and nothing here runs at all without a
terminal — rigma invokes these commands as a subprocess, and a prompt in that
position hangs a background ingest until it times out.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from rich.console import Console

from .discover import Candidate, discover
from .sources import add_sources

console = Console()

LARGE_CORPUS = 5_000     # above this, first ingest is minutes, so confirm it


def is_interactive() -> bool:
    if os.environ.get("RAGGITY_NONINTERACTIVE"):
        return False
    try:
        return bool(sys.stdin.isatty())
    except (AttributeError, ValueError):
        return False


def _read_choice(prompt: str) -> str:
    return console.input(prompt).strip().lower()


def _confirm(message: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = console.input(f"{message} {suffix} ").strip().lower()
    if not answer:
        return default
    return answer.startswith("y")


def _browse() -> Path | None:
    """The native folder dialog, or a typed path when there is no display."""
    try:
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        chosen = filedialog.askdirectory(title="Choose a folder to index")
        root.destroy()
    except Exception as exc:                     # noqa: BLE001
        console.print(f"[yellow]No folder picker available here ({exc}).[/yellow]")
        typed = console.input("Type a folder path instead (blank to cancel): ")
        return Path(typed.strip()).expanduser() if typed.strip() else None
    return Path(chosen) if chosen else None


def _render(cands: list[Candidate], complete: bool) -> None:
    console.print("\n[bold]No documents are indexed yet.[/bold] "
                  "Here is what I found on this machine:\n")
    for i, c in enumerate(cands, 1):
        console.print(f"  [cyan]{i}[/cyan]  {c.path}  [dim]{c.why}[/dim]")
    if not complete:
        console.print("\n  [dim]Stopped early - [s] searches deeper.[/dim]")


def offer(config: str | None) -> bool:
    """Returns True when sources now exist and the caller may continue."""
    if not is_interactive():
        return False
    deep = False
    scanned = False
    cands: list[Candidate] = []
    complete = True
    while True:
        # Scan once, and again only when the user asks to search deeper.
        # Re-scanning per iteration meant declining the size confirmation, or
        # mistyping a number, cost another full sweep of the disk.
        if not scanned:
            cands, complete = discover(deep=deep)
            scanned = True
        if not cands:
            console.print("[yellow]Found no obvious document folders.[/yellow]")
        else:
            _render(cands, complete)
        choice = _read_choice(
            "\n  numbers to index   [b] browse   [s] search deeper   [q] quit\n> ")
        if choice in ("q", ""):
            return False
        if choice == "s":
            deep, scanned = True, False
            continue
        if choice == "b":
            picked = _browse()
            chosen = [picked] if picked and picked.is_dir() else []
        else:
            wanted = {int(t) for t in choice.replace(",", " ").split()
                      if t.isdigit()}
            chosen = [c.path for i, c in enumerate(cands, 1) if i in wanted]
        if not chosen:
            console.print("[yellow]Nothing selected.[/yellow]")
            continue

        by_path = {c.path: c for c in cands}
        total = sum(by_path[p].file_count for p in chosen if p in by_path)
        if total > LARGE_CORPUS and not _confirm(
                f"That is {total:,} files - the first index will take minutes. "
                "Continue?", default=False):
            continue

        dest = add_sources(chosen, config)
        console.print(f"[green]Added {len(chosen)} folder(s)[/green] to {dest.name}")
        if _confirm("Index them now?", default=True):
            from .cli import _do_ingest
            _do_ingest(str(dest))
        return True
