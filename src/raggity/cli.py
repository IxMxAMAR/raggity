from __future__ import annotations

import shutil
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
        expand: bool = typer.Option(False, "--expand"),
        hyde: bool = typer.Option(False, "--hyde"),
        step_back: bool = typer.Option(False, "--step-back"),
        no_stream: bool = typer.Option(False, "--no-stream")):
    """Ask a question against your knowledge base."""
    import asyncio
    rag = _rag(config)
    if expand or hyde or step_back:
        typer.echo("Query transforms enabled (+model calls)…", err=True)
    expand_arg = True if expand else None
    hyde_arg = True if hyde else None
    step_back_arg = True if step_back else None
    if plain or no_stream:
        # Buffered path — unchanged behaviour
        answer = rag.ask(question, expand=expand_arg, hyde=hyde_arg, step_back=step_back_arg)
        if plain:
            typer.echo(answer.text)
        else:
            console.print(answer.text)
    else:
        # Streaming path — default
        async def _stream():
            final = None
            async for piece in rag.aask_stream(question, expand=expand_arg,
                                               hyde=hyde_arg, step_back=step_back_arg):
                if isinstance(piece, str):
                    print(piece, end="", flush=True)
                else:
                    final = piece
            print()
            return final
        answer = asyncio.run(_stream())
    if answer is not None and answer.citations and not plain:
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


@app.command()
def serve(config: str = typer.Option(None, "--config"),
          host: str = typer.Option("127.0.0.1", "--host"),
          port: int = typer.Option(8000, "--port")):
    """Run the local HTTP API server."""
    try:
        import uvicorn
        from .server import create_app
    except ImportError:
        console.print("[red]The server needs extra deps:[/red] pip install raggity[server]")
        raise typer.Exit(1)
    uvicorn.run(create_app(load_config(config)), host=host, port=port)


if __name__ == "__main__":
    app()
