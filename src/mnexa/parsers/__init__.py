"""Source-document parsers. Dispatch by file extension; lazy-import heavies."""

from __future__ import annotations

from pathlib import Path


def read_source(path: Path) -> str:
    """Extract plain text from a raw source document."""
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt", ""}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        from mnexa.parsers.pdf import read_pdf

        return read_pdf(path)
    if suffix == ".docx":
        from mnexa.parsers.docx import read_docx

        return read_docx(path)
    raise ValueError(
        f"unsupported file type: {suffix!r}. v0 supports .md, .txt, .pdf, .docx."
    )
