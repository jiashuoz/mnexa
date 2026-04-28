# pyright: basic
"""Thin REST wrapper around Granola's `/v1/notes` API.

Two operations: list notes (paginated, optionally `created_after`) and
fetch one note (with transcript). Lazy-imports `httpx` so non-Granola
callers don't pay the import cost.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

API_BASE = "https://api.granola.ai/v1"


@dataclass
class GranolaNote:
    note_id: str
    title: str
    summary: str
    transcript: list[dict[str, Any]]
    created_at: str
    modified_at: str
    owner_name: str | None
    owner_email: str | None
    participants: list[str]
    raw: dict[str, Any]  # everything else, in case we add fields later

    @property
    def url(self) -> str:
        return f"https://app.granola.ai/notes/{self.note_id}"


@dataclass
class GranolaNoteSummary:
    """List-endpoint shape — no transcript, just enough to decide if we ingest."""
    note_id: str
    title: str
    created_at: str
    modified_at: str


class GranolaClient:
    def __init__(self, api_key: str) -> None:
        import httpx

        self._client = httpx.Client(
            base_url=API_BASE,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )

    def list_notes(
        self, *, created_after: str | None = None, page_size: int = 100,
    ) -> Iterator[GranolaNoteSummary]:
        """Yield all notes, paginating via cursor."""
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"limit": page_size}
            if cursor:
                params["cursor"] = cursor
            if created_after:
                params["created_after"] = created_after
            resp = self._client.get("/notes", params=params)
            resp.raise_for_status()
            data = resp.json()
            for raw in data.get("notes", []) or data.get("data", []) or []:
                yield _to_note_summary(raw)
            cursor = data.get("next_cursor") or data.get("cursor")
            if not cursor:
                break

    def get_note(self, note_id: str) -> GranolaNote:
        resp = self._client.get(
            f"/notes/{note_id}", params={"include": "transcript"},
        )
        resp.raise_for_status()
        return _to_note(resp.json())

    def close(self) -> None:
        self._client.close()


def _to_note_summary(raw: dict[str, Any]) -> GranolaNoteSummary:
    return GranolaNoteSummary(
        note_id=raw["id"],
        title=raw.get("title") or "(untitled)",
        created_at=raw.get("created_at") or raw.get("createdAt") or "",
        modified_at=raw.get("modified_at") or raw.get("modifiedAt") or "",
    )


def _to_note(raw: dict[str, Any]) -> GranolaNote:
    owner = raw.get("owner") or {}
    participants_raw = raw.get("participants") or []
    participants: list[str] = []
    for p in participants_raw:
        if isinstance(p, str):
            participants.append(p)
        elif isinstance(p, dict):
            name = p.get("name") or p.get("email")
            if name:
                participants.append(name)
    return GranolaNote(
        note_id=raw["id"],
        title=raw.get("title") or "(untitled)",
        summary=raw.get("summary") or "",
        transcript=raw.get("transcript") or [],
        created_at=raw.get("created_at") or raw.get("createdAt") or "",
        modified_at=raw.get("modified_at") or raw.get("modifiedAt") or "",
        owner_name=(owner.get("name") if isinstance(owner, dict) else None),
        owner_email=(owner.get("email") if isinstance(owner, dict) else None),
        participants=participants,
        raw=raw,
    )


def render_note_text(note: GranolaNote) -> str:
    """Flatten a GranolaNote into the plain text we feed to the LLM."""
    parts: list[str] = []
    parts.append(f"# {note.title}\n")
    if note.owner_name or note.owner_email:
        owner = note.owner_name or note.owner_email
        parts.append(f"Owner: {owner}")
    if note.participants:
        parts.append("Participants: " + ", ".join(note.participants))
    if note.created_at:
        parts.append(f"Created: {note.created_at}")
    parts.append("")
    if note.summary:
        parts.append("## Summary\n")
        parts.append(note.summary)
        parts.append("")
    if note.transcript:
        parts.append("## Transcript\n")
        for turn in note.transcript:
            speaker = (
                turn.get("speaker")
                or turn.get("diarization_label")
                or "speaker"
            )
            text = turn.get("text") or ""
            if text:
                parts.append(f"{speaker}: {text}")
    return "\n".join(parts)
