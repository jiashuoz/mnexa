from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from mnexa import ingest as ingest_mod
from mnexa import lint as lint_mod
from mnexa import query as query_mod
from mnexa import storage

# Load .env from cwd (or any parent dir) before any command runs.
load_dotenv()

app = typer.Typer(
    name="mnexa",
    help="A disciplined wiki maintainer for a personal markdown knowledge base.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def init(
    path: Path = typer.Argument(..., help="Where to create the new vault."),
) -> None:
    """Create a new vault at PATH."""
    storage.init_vault(path)


@app.command()
def ingest(
    target: str = typer.Argument(
        ...,
        help="A local file, local folder, Drive file URL, or Drive folder URL.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation prompts on folder ingests.",
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Max files to ingest (folder mode).",
    ),
) -> None:
    """Ingest a file, folder, or Drive URL into the wiki."""
    ingest_mod.run(target, yes=yes, limit=limit)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask the wiki."),
) -> None:
    """Ask the wiki a question."""
    query_mod.run(question)


@app.command()
def lint(
    fix: bool = typer.Option(False, "--fix", help="Interactively fix lint findings."),
) -> None:
    """Audit the wiki for issues."""
    lint_mod.run(fix=fix)


if __name__ == "__main__":
    app()
