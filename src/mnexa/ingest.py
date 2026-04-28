"""Two-stage ingest pipeline (analyze → generate) over any source.

A source is a local file, a Drive file, a local folder, or a Drive folder.
The user types `mnexa ingest <anything>` and we dispatch.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import shutil
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import typer

if TYPE_CHECKING:
    from mnexa.drive.client import DriveClient, DriveFile
    from mnexa.granola.client import GranolaClient, GranolaNoteSummary

from mnexa import storage
from mnexa.llm import LLMClient, Usage, get_client
from mnexa.parser import parse_file_blocks, verify_grounding
from mnexa.parsers import read_source
from mnexa.prompts import load as load_prompt

MAX_SOURCE_BYTES = 200_000
MAX_RELATED_PAGES = 10
FOLDER_CONFIRM_THRESHOLD = 5


# --- target classification & data shapes -----------------------------------


TargetKind = Literal[
    "local-file", "local-folder",
    "drive-file", "drive-folder",
    "granola-note", "granola-list",
]


@dataclass(frozen=True)
class IngestTarget:
    kind: TargetKind
    local_path: Path | None = None
    external_id: str | None = None
    since: str | None = None  # only used by granola-list


@dataclass(frozen=True)
class DriveMeta:
    file_id: str
    modified_time: str
    web_view_link: str
    drive_path: str
    mime_type: str


@dataclass(frozen=True)
class GranolaMeta:
    note_id: str
    created_at: str
    modified_at: str
    web_view_link: str
    participants: list[str]


@dataclass(frozen=True)
class IngestSource:
    filename: str       # display name, e.g., "foo.pdf" or "Design review"
    text: str           # parsed plain text fed to the LLM
    hash: str           # sha256 of raw bytes (for change detection)
    source_path: str    # frontmatter source_path: "raw/foo.pdf" / "drive://<id>" / "granola://<id>"
    drive_meta: DriveMeta | None = None
    granola_meta: GranolaMeta | None = None


_DRIVE_FOLDER_RE = re.compile(r"/folders/([a-zA-Z0-9_-]{8,})")
_DRIVE_FILE_RE = re.compile(r"/file/d/([a-zA-Z0-9_-]{8,})")
_DRIVE_QUERY_ID_RE = re.compile(r"[?&]id=([a-zA-Z0-9_-]{8,})")
_DRIVE_SCHEME_RE = re.compile(r"^drive://([a-zA-Z0-9_-]{8,})")

_GRANOLA_NOTE_URL_RE = re.compile(r"app\.granola\.ai/notes/([a-zA-Z0-9_-]{8,})")
_GRANOLA_NOTE_SCHEME_RE = re.compile(r"^granola://note/([a-zA-Z0-9_-]{8,})")
_GRANOLA_SINCE_RE = re.compile(r"^granola://since/(.+)$")


def classify_target(arg: str) -> IngestTarget:
    arg = arg.strip()

    # Drive
    if "drive.google.com" in arg or arg.startswith("drive://"):
        if "/folders/" in arg:
            return IngestTarget(
                "drive-folder", external_id=_extract_drive_folder_id(arg)
            )
        return IngestTarget("drive-file", external_id=_extract_drive_file_id(arg))

    # Granola
    if "granola.ai" in arg or arg.startswith("granola"):
        for pattern in (_GRANOLA_NOTE_URL_RE, _GRANOLA_NOTE_SCHEME_RE):
            m = pattern.search(arg)
            if m:
                return IngestTarget("granola-note", external_id=m.group(1))
        m = _GRANOLA_SINCE_RE.search(arg)
        if m:
            return IngestTarget("granola-list", since=m.group(1))
        if arg in {"granola", "granola://", "granola://recent"}:
            return IngestTarget("granola-list")
        raise typer.BadParameter(f"unrecognized Granola argument: {arg!r}")

    # Local
    p = Path(arg).expanduser()
    if p.is_dir():
        return IngestTarget("local-folder", local_path=p.resolve())
    if p.is_file():
        return IngestTarget("local-file", local_path=p.resolve())
    raise typer.BadParameter(
        f"can't interpret {arg!r} as a file, folder, or supported URL"
    )


def _extract_drive_folder_id(url: str) -> str:
    m = _DRIVE_FOLDER_RE.search(url)
    if not m:
        raise typer.BadParameter(f"could not parse Drive folder ID from {url!r}")
    return m.group(1)


def _extract_drive_file_id(url: str) -> str:
    for pattern in (_DRIVE_FILE_RE, _DRIVE_QUERY_ID_RE, _DRIVE_SCHEME_RE):
        m = pattern.search(url)
        if m:
            return m.group(1)
    raise typer.BadParameter(f"could not parse Drive file ID from {url!r}")


# --- entry point ------------------------------------------------------------


def run(target: str | Path, *, client: LLMClient | None = None,
        yes: bool = False, limit: int | None = None,
        since: str | None = None) -> None:
    asyncio.run(_run_async(
        str(target), client=client, yes=yes, limit=limit, since=since,
    ))


async def _run_async(target: str, *, client: LLMClient | None,
                     yes: bool, limit: int | None, since: str | None) -> None:
    vault = storage.find_vault(Path.cwd())
    if vault is None:
        typer.echo(
            "error: not inside an Mnexa vault (run `mnexa init` first)", err=True
        )
        raise typer.Exit(1)

    tgt = classify_target(target)

    if client is None:
        client = get_client()

    if tgt.kind == "local-file":
        assert tgt.local_path is not None
        source = _load_local_source(tgt.local_path, vault)
        await _ingest_one(source, vault=vault, client=client)
        return

    if tgt.kind == "local-folder":
        assert tgt.local_path is not None
        await _ingest_local_folder(tgt.local_path, vault=vault, client=client,
                                   yes=yes, limit=limit)
        return

    if tgt.kind == "drive-file":
        assert tgt.external_id is not None
        from mnexa.drive.auth import get_credentials
        from mnexa.drive.client import DriveClient
        creds = get_credentials()
        drive_client = DriveClient(creds)
        source = _load_drive_source(tgt.external_id, drive_client)
        await _ingest_one(source, vault=vault, client=client)
        return

    if tgt.kind == "drive-folder":
        assert tgt.external_id is not None
        from mnexa.drive.auth import get_credentials
        from mnexa.drive.client import DriveClient
        creds = get_credentials()
        drive_client = DriveClient(creds)
        await _ingest_drive_folder(
            tgt.external_id, vault=vault, client=client,
            drive_client=drive_client, yes=yes, limit=limit,
        )
        return

    if tgt.kind == "granola-note":
        assert tgt.external_id is not None
        from mnexa.granola.auth import get_api_key
        from mnexa.granola.client import GranolaClient
        gc = GranolaClient(get_api_key())
        try:
            source = _load_granola_source(tgt.external_id, gc)
            await _ingest_one(source, vault=vault, client=client)
        finally:
            gc.close()
        return

    if tgt.kind == "granola-list":
        from mnexa.granola.auth import get_api_key
        from mnexa.granola.client import GranolaClient
        gc = GranolaClient(get_api_key())
        try:
            await _ingest_granola_list(
                vault=vault, client=client, granola_client=gc,
                yes=yes, limit=limit, since=since or tgt.since,
            )
        finally:
            gc.close()
        return


# --- folder ingest ----------------------------------------------------------


@dataclass(frozen=True)
class _ExistingExternalPage:
    path: Path
    modified: str | None
    hash: str | None


def _existing_external_pages(
    vault: Path, *, id_field: str, mtime_field: str,
) -> dict[str, _ExistingExternalPage]:
    """Map external-id → existing wiki page metadata, by reading frontmatter.

    Generic over the id field (`drive_file_id`, `granola_note_id`, etc.) and
    the corresponding modified-time field.
    """
    out: dict[str, _ExistingExternalPage] = {}
    sources = vault / "wiki" / "sources"
    if not sources.is_dir():
        return out
    for p in sources.glob("*.md"):
        fm = storage.read_frontmatter(p)
        ident = fm.get(id_field)
        if isinstance(ident, str) and ident:
            mod = _normalize_timestamp(fm.get(mtime_field))
            h = fm.get("hash")
            out[ident] = _ExistingExternalPage(
                path=p,
                modified=mod,
                hash=h if isinstance(h, str) else None,
            )
    return out


def _normalize_timestamp(value: object) -> str | None:
    """Coerce a frontmatter timestamp value to canonical RFC 3339 string.

    YAML auto-parses unquoted ISO-8601 timestamps to `datetime`, which breaks
    string equality checks against Drive's `modifiedTime` (which is always a
    string from the API). Normalise both to the same string form.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        ms = value.microsecond // 1000
        return value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms:03d}Z"
    return None


async def _ingest_drive_folder(folder_id: str, *, vault: Path, client: LLMClient,
                               drive_client: DriveClient, yes: bool,
                               limit: int | None) -> None:
    typer.echo(f"[ingest] scanning Drive folder {folder_id}…", err=True)
    items = list(drive_client.walk(folder_id))
    if not items:
        typer.echo("[ingest] folder is empty", err=True)
        return

    existing = _existing_external_pages(
        vault, id_field="drive_file_id", mtime_field="drive_modified",
    )
    pending: list[tuple[str, DriveFile]] = []
    skipped = 0
    for drive_path, df in items:
        prev = existing.get(df.file_id)
        if prev is not None and prev.modified == df.modified_time:
            skipped += 1
            continue
        pending.append((drive_path, df))

    if limit is not None:
        pending = pending[:limit]

    typer.echo(
        f"[ingest] {len(items)} files in folder · {len(pending)} new/changed · "
        f"{skipped} unchanged",
        err=True,
    )
    if not pending:
        return

    if (
        not yes and len(pending) >= FOLDER_CONFIRM_THRESHOLD
        and not typer.confirm(f"proceed with {len(pending)} ingests?")
    ):
        typer.echo("[ingest] aborted", err=True)
        return

    succeeded = 0
    failed = 0
    for i, (drive_path, df) in enumerate(pending, 1):
        typer.echo(f"[{i}/{len(pending)}] {drive_path}", err=True)
        try:
            source = _load_drive_source(df.file_id, drive_client)
            await _ingest_one(source, vault=vault, client=client)
            succeeded += 1
        except (RuntimeError, OSError, ValueError) as e:
            typer.echo(f"  failed: {e}", err=True)
            failed += 1
            continue

    typer.echo(
        f"[ingest] folder done · {succeeded} ingested · {failed} failed", err=True
    )


async def _ingest_granola_list(*, vault: Path, client: LLMClient,
                               granola_client: GranolaClient, yes: bool,
                               limit: int | None, since: str | None) -> None:
    typer.echo("[ingest] listing Granola notes…", err=True)
    summaries = list(granola_client.list_notes(created_after=since))
    if not summaries:
        typer.echo("[ingest] no notes returned", err=True)
        return

    existing = _existing_external_pages(
        vault, id_field="granola_note_id", mtime_field="granola_modified",
    )
    pending: list[GranolaNoteSummary] = []
    skipped = 0
    for s in summaries:
        prev = existing.get(s.note_id)
        if prev is not None and prev.modified == s.modified_at:
            skipped += 1
            continue
        pending.append(s)

    if limit is not None:
        pending = pending[:limit]

    typer.echo(
        f"[ingest] {len(summaries)} notes total · {len(pending)} new/changed · "
        f"{skipped} unchanged",
        err=True,
    )
    if not pending:
        return

    if (
        not yes and len(pending) >= FOLDER_CONFIRM_THRESHOLD
        and not typer.confirm(f"proceed with {len(pending)} ingests?")
    ):
        typer.echo("[ingest] aborted", err=True)
        return

    succeeded = 0
    failed = 0
    for i, s in enumerate(pending, 1):
        typer.echo(f"[{i}/{len(pending)}] {s.title}", err=True)
        try:
            source = _load_granola_source(s.note_id, granola_client)
            await _ingest_one(source, vault=vault, client=client)
            succeeded += 1
        except (RuntimeError, OSError, ValueError) as e:
            typer.echo(f"  failed: {e}", err=True)
            failed += 1
            continue

    typer.echo(
        f"[ingest] granola list done · {succeeded} ingested · {failed} failed",
        err=True,
    )


async def _ingest_local_folder(folder: Path, *, vault: Path, client: LLMClient,
                               yes: bool, limit: int | None) -> None:
    typer.echo(f"[ingest] scanning local folder {folder}…", err=True)
    files: list[Path] = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in {".md", ".markdown", ".txt", ".pdf", ".docx"}
    )
    if limit is not None:
        files = files[:limit]
    if not files:
        typer.echo("[ingest] no supported files found", err=True)
        return

    typer.echo(f"[ingest] {len(files)} files to consider", err=True)
    if (
        not yes and len(files) >= FOLDER_CONFIRM_THRESHOLD
        and not typer.confirm(f"proceed with {len(files)} ingests?")
    ):
        typer.echo("[ingest] aborted", err=True)
        return

    succeeded = 0
    failed = 0
    for i, file in enumerate(files, 1):
        typer.echo(f"[{i}/{len(files)}] {file.relative_to(folder)}", err=True)
        try:
            source = _load_local_source(file, vault)
            await _ingest_one(source, vault=vault, client=client)
            succeeded += 1
        except (RuntimeError, OSError, ValueError) as e:
            typer.echo(f"  failed: {e}", err=True)
            failed += 1
            continue

    typer.echo(
        f"[ingest] folder done · {succeeded} ingested · {failed} failed", err=True
    )


# --- source loaders ---------------------------------------------------------


def _load_local_source(file: Path, vault: Path) -> IngestSource:
    file = file.expanduser().resolve()
    if not file.is_file():
        raise ValueError(f"not a file: {file}")
    raw_bytes = file.read_bytes()
    if len(raw_bytes) > MAX_SOURCE_BYTES:
        raise ValueError(
            f"source is {len(raw_bytes)} bytes; v0 limit is {MAX_SOURCE_BYTES}"
        )
    text = read_source(file)
    raw_dest = vault / "raw" / file.name
    if file.parent.resolve() != (vault / "raw").resolve() and not raw_dest.exists():
        shutil.copy2(file, raw_dest)
    return IngestSource(
        filename=file.name,
        text=text,
        hash=hashlib.sha256(raw_bytes).hexdigest(),
        source_path=f"raw/{file.name}",
        drive_meta=None,
    )


def _load_granola_source(note_id: str, granola_client: GranolaClient) -> IngestSource:
    from mnexa.granola.client import render_note_text
    note = granola_client.get_note(note_id)
    text = render_note_text(note)
    if len(text.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError(
            f"note {note_id} text is {len(text.encode('utf-8'))} bytes; "
            f"v0 limit is {MAX_SOURCE_BYTES}"
        )
    return IngestSource(
        filename=note.title,
        text=text,
        hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        source_path=f"granola://{note.note_id}",
        drive_meta=None,
        granola_meta=GranolaMeta(
            note_id=note.note_id,
            created_at=note.created_at,
            modified_at=note.modified_at,
            web_view_link=note.url,
            participants=note.participants,
        ),
    )


def _load_drive_source(file_id: str, drive_client: DriveClient) -> IngestSource:
    df = drive_client.get(file_id)
    if df.is_folder:
        raise ValueError(f"{file_id} is a folder, not a file")
    content_bytes, ext = drive_client.download(df)
    if len(content_bytes) > MAX_SOURCE_BYTES:
        raise ValueError(
            f"source is {len(content_bytes)} bytes; v0 limit is {MAX_SOURCE_BYTES}"
        )
    filename = df.name + ext if ext and not df.name.endswith(ext) else df.name
    text = _bytes_to_text(content_bytes, df.mime_type, filename)
    return IngestSource(
        filename=filename,
        text=text,
        hash=hashlib.sha256(content_bytes).hexdigest(),
        source_path=f"drive://{df.file_id}",
        drive_meta=DriveMeta(
            file_id=df.file_id,
            modified_time=df.modified_time,
            web_view_link=f"https://drive.google.com/file/d/{df.file_id}/view",
            drive_path=df.name,
            mime_type=df.mime_type,
        ),
    )


def _bytes_to_text(content: bytes, mime_type: str, filename: str) -> str:
    if mime_type in {"text/plain", "text/markdown"}:
        return content.decode("utf-8", errors="replace")
    suffix = Path(filename).suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(content)
        tmp_path = Path(tf.name)
    try:
        return read_source(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


# --- per-source pipeline ----------------------------------------------------


async def _ingest_one(source: IngestSource, *, vault: Path, client: LLMClient) -> None:
    schema = (vault / "CLAUDE.md").read_text(encoding="utf-8")
    index = (vault / "wiki" / "index.md").read_text(encoding="utf-8")
    related = _find_related_pages(vault, source.text, MAX_RELATED_PAGES)
    today = date.today().isoformat()

    typer.echo(f"  [stage 1] analyzing {source.filename}…", err=True)
    stage1_system = _build_system("stage1.md", schema)
    stage1_user = _build_stage1_user(
        index=index, related=related, vault=vault, source=source,
    )
    completion = await client.complete(
        system=stage1_system, user=stage1_user, cache_system=True
    )
    analysis = completion.text
    typer.echo(f"  [stage 1] done · {_fmt_usage(completion.usage)}", err=True)

    typer.echo("  [stage 2] generating wiki updates…", err=True)
    existing = _gather_existing_pages(vault, analysis)
    stage2_system = _build_system("stage2.md", schema)
    stage2_user = _build_stage2_user(
        analysis=analysis, vault=vault, source=source,
        existing=existing, today=today,
    )

    accumulated: list[str] = []
    async for chunk in client.stream(
        system=stage2_system, user=stage2_user, cache_system=True
    ):
        sys.stderr.write(chunk)
        sys.stderr.flush()
        accumulated.append(chunk)
    sys.stderr.write("\n")
    output = "".join(accumulated)
    if client.last_usage is not None:
        typer.echo(f"  [stage 2] done · {_fmt_usage(client.last_usage)}", err=True)

    blocks = parse_file_blocks(output, vault)
    if not blocks:
        typer.echo("  no changes (Stage 2 emitted no FILE blocks)", err=True)
        return

    verify_grounding(blocks, source.text)

    pages = {b.abs_path: b.raw_content for b in blocks}
    try:
        storage.write_pages(vault, pages)
    except Exception:
        storage.git_rollback(vault)
        raise

    if not storage.git_commit(vault, f"ingest: {source.filename}"):
        typer.echo("  warning: write succeeded but no git changes detected", err=True)
        return


# --- helpers ----------------------------------------------------------------


_TOKEN_RE = re.compile(r"\w+")
_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "had", "has", "have", "her", "his", "its", "may", "one", "our", "out",
    "she", "two", "way", "who", "with", "this", "that", "from", "they",
    "them", "their", "there", "what", "when", "where", "which", "while",
    "would", "could", "should", "than", "then", "into", "your",
})


def _tokens(s: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(s.lower()) if len(t) > 2 and t not in _STOPWORDS}


def _find_related_pages(vault: Path, source_text: str, top_n: int) -> list[Path]:
    src_tokens = _tokens(source_text)
    if not src_tokens:
        return []
    wiki = vault / "wiki"
    scored: list[tuple[int, Path]] = []
    for p in wiki.rglob("*.md"):
        if p.name in {"index.md", "log.md"}:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        score = len(src_tokens & _tokens(text))
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:top_n]]


def _gather_existing_pages(vault: Path, analysis: str) -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for m in re.finditer(r"wiki/[\w/\-.]+\.md", analysis):
        p = (vault / m.group(0)).resolve()
        if p in seen:
            continue
        seen.add(p)
        if p.is_file():
            paths.append(p)
    for name in ("index.md", "log.md"):
        p = (vault / "wiki" / name).resolve()
        if p not in seen and p.is_file():
            seen.add(p)
            paths.append(p)
    return paths


def _build_system(prompt_name: str, schema: str) -> str:
    return f"{load_prompt(prompt_name)}\n\n<schema>\n{schema}\n</schema>"


def _read_pages(paths: Iterable[Path], vault: Path) -> str:
    parts: list[str] = []
    for p in paths:
        rel = p.relative_to(vault)
        parts.append(f"--- {rel} ---\n{p.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)


def _drive_meta_block(meta: DriveMeta) -> str:
    return (
        "<drive_meta>\n"
        f"file_id: {meta.file_id}\n"
        f"modified_time: {meta.modified_time}\n"
        f"web_view_link: {meta.web_view_link}\n"
        f"drive_path: {meta.drive_path}\n"
        f"mime_type: {meta.mime_type}\n"
        "</drive_meta>"
    )


def _granola_meta_block(meta: GranolaMeta) -> str:
    participants = ", ".join(meta.participants) if meta.participants else ""
    return (
        "<granola_meta>\n"
        f"note_id: {meta.note_id}\n"
        f"created_at: {meta.created_at}\n"
        f"modified_at: {meta.modified_at}\n"
        f"web_view_link: {meta.web_view_link}\n"
        f"participants: {participants}\n"
        "</granola_meta>"
    )


def _external_meta_block(source: IngestSource) -> str:
    if source.drive_meta is not None:
        return _drive_meta_block(source.drive_meta)
    if source.granola_meta is not None:
        return _granola_meta_block(source.granola_meta)
    return ""


def _build_stage1_user(
    *, index: str, related: list[Path], vault: Path, source: IngestSource
) -> str:
    related_block = _read_pages(related, vault) if related else "(none)"
    meta_block = _external_meta_block(source)
    return (
        f"<index>\n{index}\n</index>\n\n"
        f"<related_pages>\n{related_block}\n</related_pages>\n\n"
        f'<source filename="{source.filename}">\n{source.text}\n</source>'
        + (f"\n\n{meta_block}" if meta_block else "")
    )


def _build_stage2_user(
    *, analysis: str, vault: Path, source: IngestSource,
    existing: list[Path], today: str,
) -> str:
    existing_block = _read_pages(existing, vault) if existing else "(none)"
    meta_block = _external_meta_block(source)
    return (
        f"<analysis>\n{analysis}\n</analysis>\n\n"
        f'<source filename="{source.filename}" hash="{source.hash}" '
        f'source_path="{source.source_path}">\n{source.text}\n</source>'
        + (f"\n\n{meta_block}" if meta_block else "")
        + f"\n\n<existing_pages>\n{existing_block}\n</existing_pages>\n\n"
        f"<today>{today}</today>"
    )


def _fmt_usage(u: Usage) -> str:
    return f"in={u.input_tokens} out={u.output_tokens} cached={u.cached_input_tokens}"
