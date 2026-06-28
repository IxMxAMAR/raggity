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


@app.command(name="ingest-url")
def ingest_url(
    url: str = typer.Argument(..., help="URL to fetch and index."),
    depth: int = typer.Option(0, "--depth", "-d", help="BFS crawl depth (0 = start URL only)."),
    config: str = typer.Option(None, "--config"),
):
    """Fetch a URL (and optionally crawl same-domain links) and add to the index."""
    try:
        from .connectors.web import WebConnector  # noqa: PLC0415
    except ImportError:
        console.print("[red]ingest-url needs extra deps:[/red] pip install raggity[web]")
        raise typer.Exit(1)
    rag = _rag(config)
    console.print(f"Fetching [cyan]{url}[/cyan] (depth={depth})…")
    try:
        connector = WebConnector(url, depth=depth, same_domain=True)
        docs = connector.fetch()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    if not docs:
        console.print("[yellow]No content extracted.[/yellow]")
        raise typer.Exit(0)
    added = rag.ingest_documents(docs)
    console.print(f"[green]Ingested {added} page(s) from {url}.[/green]")


@app.command(name="ingest-repo")
def ingest_repo(
    url: str = typer.Argument(..., help="Git repository URL to clone and index."),
    ref: str = typer.Option(None, "--ref", "-r", help="Branch or tag to check out (default: HEAD)."),
    config: str = typer.Option(None, "--config"),
):
    """Shallow-clone a git repository and add all text files to the index."""
    from .connectors.github import GitHubConnector  # noqa: PLC0415
    rag = _rag(config)
    console.print(f"Cloning [cyan]{url}[/cyan]" + (f" @ {ref}" if ref else "") + "…")
    try:
        connector = GitHubConnector(url, ref=ref)
        docs = connector.fetch()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    if not docs:
        console.print("[yellow]No text files found in repository.[/yellow]")
        raise typer.Exit(0)
    added = rag.ingest_documents(docs)
    console.print(f"[green]Ingested {added} file(s) from {url}.[/green]")


@app.command(name="ingest-obsidian")
def ingest_obsidian(
    vault: str = typer.Argument(..., help="Path to the Obsidian vault directory."),
    config: str = typer.Option(None, "--config"),
):
    """Read all Markdown notes from an Obsidian vault and add them to the index."""
    from .connectors.obsidian import ObsidianConnector  # noqa: PLC0415
    rag = _rag(config)
    console.print(f"Reading vault [cyan]{vault}[/cyan]…")
    try:
        connector = ObsidianConnector(vault)
        docs = connector.fetch()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    if not docs:
        console.print("[yellow]No Markdown notes found in vault.[/yellow]")
        raise typer.Exit(0)
    added = rag.ingest_documents(docs)
    console.print(f"[green]Ingested {added} note(s) from {vault}.[/green]")


@app.command()
def ask(question: str, config: str = typer.Option(None, "--config"),
        plain: bool = typer.Option(False, "--plain"),
        expand: bool = typer.Option(False, "--expand"),
        hyde: bool = typer.Option(False, "--hyde"),
        step_back: bool = typer.Option(False, "--step-back"),
        no_stream: bool = typer.Option(False, "--no-stream"),
        decompose: bool = typer.Option(False, "--decompose"),
        no_cache: bool = typer.Option(False, "--no-cache")):
    """Ask a question against your knowledge base."""
    import asyncio
    rag = _rag(config)
    if decompose:
        if expand or hyde or step_back:
            typer.echo("note: --decompose overrides other query transforms", err=True)
        typer.echo("Decomposing query (+model calls)…", err=True)
        answer = rag.ask_decompose(question)
        if plain:
            typer.echo(answer.text)
        else:
            console.print(answer.text)
    else:
        if expand or hyde or step_back:
            typer.echo("Query transforms enabled (+model calls)…", err=True)
        expand_arg = True if expand else None
        hyde_arg = True if hyde else None
        step_back_arg = True if step_back else None
        use_cache_arg = False if no_cache else None
        if plain or no_stream:
            # Buffered path — honors cache (unless --no-cache)
            answer = rag.ask(question, expand=expand_arg, hyde=hyde_arg, step_back=step_back_arg,
                             use_cache=use_cache_arg)
            if plain:
                typer.echo(answer.text)
            else:
                console.print(answer.text)
        else:
            # Streaming path — default; always calls the model (cache is buffered-only)
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
             k: int = typer.Option(5, "--k"),
             llm_judge: bool = typer.Option(False, "--llm-judge")):
    """Run free CPU retrieval metrics against a golden.jsonl set."""
    rag = _rag(config)
    if llm_judge:
        typer.echo("Running LLM-judge eval (+2 model calls per question)…", err=True)
        import asyncio
        from .evaluate import llm_judge as run_judge
        res = asyncio.run(run_judge(rag, load_golden(golden), rag.provider))
        console.print(f"Faithfulness={res.faithfulness:.3f}  "
                      f"AnswerRelevance={res.answer_relevance:.3f}  (n={res.n})")
        console.print("(note: self-assessed — same model family generates and grades)")
    else:
        res = evaluate(rag.retriever, load_golden(golden), k=k)
        console.print(f"Hit@{k}={res.hit_rate:.3f}  MRR={res.mrr:.3f}  "
                      f"Recall@{k}={res.recall:.3f}  (n={res.n})")


@app.command()
def watch(config: str = typer.Option(None, "--config"),
          debounce: float = typer.Option(2.0, "--debounce")):
    """Watch source folders and re-index on change (Ctrl-C to stop)."""
    rag = _rag(config)
    try:
        from .watch import run_watch
    except ImportError:
        console.print("[red]watch needs extra deps:[/red] pip install raggity[watch]")
        raise typer.Exit(1)
    try:
        observer = run_watch(rag, rag.cfg.sources.include, debounce)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Watching[/green] {len(rag.cfg.sources.include)} source pattern(s). Ctrl-C to stop.")
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


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
