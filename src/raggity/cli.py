from __future__ import annotations

import typer
from rich.console import Console

from .config import load_config
from .core import Raggity
from .evaluate import evaluate, load_golden

app = typer.Typer(help="raggity — local-first RAG over your notes, answered by Claude.")
console = Console()


def _rag(config: str | None) -> Raggity:
    return Raggity(load_config(config))


@app.command()
def ingest(config: str = typer.Option(None, "--config")):
    """Incrementally index configured source folders."""
    report = _rag(config).ingest()
    console.print(
        f"[green]Indexed.[/green] added={report.added} updated={report.updated} "
        f"deleted={report.deleted} unchanged={report.unchanged}"
    )


@app.command()
def ask(question: str, config: str = typer.Option(None, "--config"),
        plain: bool = typer.Option(False, "--plain"),
        expand: bool = typer.Option(False, "--expand")):
    """Ask a question against your knowledge base."""
    rag = _rag(config)
    if expand:
        typer.echo(f"Expanding query (+{rag.cfg.retrieval.expand_n} model calls)…", err=True)
    answer = rag.ask(question, expand=True if expand else None)
    if plain:
        typer.echo(answer.text)
    else:
        console.print(answer.text)
    if answer.citations and not plain:
        console.print("\n[dim]Sources:[/dim]")
        seen = set()
        for c in answer.citations:
            if c.supported and c.source_path not in seen:
                seen.add(c.source_path)
                console.print(f"  [dim]- {c.source_path}[/dim]")


@app.command()
def status(config: str = typer.Option(None, "--config")):
    """Show index statistics."""
    st = _rag(config).status()
    for k, v in st.items():
        console.print(f"{k}: {v}")


@app.command()
def reindex(config: str = typer.Option(None, "--config"),
            force: bool = typer.Option(False, "--force")):
    """Rebuild the index from scratch."""
    import shutil
    cfg = load_config(config)
    if force:
        shutil.rmtree(cfg.index.path, ignore_errors=True)
    report = Raggity(cfg).ingest()
    console.print(f"[green]Reindexed.[/green] added={report.added}")


@app.command(name="eval")
def eval_cmd(golden: str = typer.Argument(...),
             config: str = typer.Option(None, "--config"),
             k: int = typer.Option(5, "--k")):
    """Run free CPU retrieval metrics against a golden.jsonl set."""
    rag = _rag(config)
    res = evaluate(rag.retriever, load_golden(golden), k=k)
    console.print(f"Hit@{k}={res.hit_rate:.3f}  MRR={res.mrr:.3f}  "
                  f"Recall@{k}={res.recall:.3f}  (n={res.n})")


if __name__ == "__main__":
    app()
