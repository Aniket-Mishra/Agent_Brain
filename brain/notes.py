"""Reading the vault's markdown: what a note contains, and how it is split up."""

from __future__ import annotations

import hashlib
import os
import re
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

SKIPPED_DIRECTORIES = {".obsidian", ".trash", ".git", ".github", "node_modules", "__pycache__"}
FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
FENCED_CODE = re.compile(r"^[ \t]{0,3}(```|~~~).*?^[ \t]{0,3}\1", re.DOTALL | re.MULTILINE)
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
WIKILINK = re.compile(r"!?\[\[([^\[\]\n|#]+)")
HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
SPELLED_DATE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\s+(\d{4})\b")
MONTHS = {
    name: number
    for number, name in enumerate(
        "jan feb mar apr may jun jul aug sep oct nov dec".split(), start=1
    )
}


@dataclass(frozen=True)
class Note:
    path: str
    title: str
    body: str
    links: list[str]
    digest: str
    size: int = 0
    kind: str = ""
    tags: list[str] = field(default_factory=list)
    created: str = ""
    body_start_line: int = 1


@dataclass(frozen=True)
class Chunk:
    path: str
    heading: str
    text: str
    start_line: int = 0
    end_line: int = 0


def embedding_text(title: str, heading: str, text: str) -> str:
    """Breadcrumbs first, then the passage.

    "use port 5432" is ambiguous on its own. The same text under
    "Postgres > Local setup" is not, and the model can only use what it is shown.
    """
    crumbs = " > ".join(part for part in (title, heading) if part)
    return f"{crumbs}\n\n{text}"


def find_notes(vault: Path) -> list[Path]:
    root = vault.resolve()
    notes = []
    for directory, subdirectories, filenames in os.walk(vault):
        subdirectories[:] = [
            name
            for name in subdirectories
            if name not in SKIPPED_DIRECTORIES and not name.startswith(".")
        ]
        notes.extend(
            path for name in filenames
            if name.endswith(".md")
            and (path := Path(directory) / name).resolve().is_relative_to(root)
        )
    return sorted(notes)


def read_note(path: Path, vault: Path) -> Note:
    if not path.resolve().is_relative_to(vault.resolve()):
        raise ValueError("Note path resolves outside the vault")
    raw = path.read_text(encoding="utf-8", errors="replace")
    match = FRONTMATTER.match(raw)
    body = raw[match.end():] if match else raw
    fields = read_frontmatter(match.group(1) if match else "")

    return Note(
        path=path.relative_to(vault).as_posix(),
        title=read_title(fields, path.stem),
        body=body,
        links=find_links(body),
        digest=hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest(),
        size=len(raw),
        kind=str(fields.get("type") or "").strip(),
        tags=as_list(fields.get("tags")),
        created=read_date(fields),
        body_start_line=raw.count("\n", 0, match.end()) + 1 if match else 1,
    )


def read_title(fields: dict, fallback: str) -> str:
    """Notes here use `note_title`; `title` is the more common convention."""
    for key in ("note_title", "title"):
        declared = str(fields.get(key) or "").strip()
        if declared:
            return declared
    return fallback


def read_frontmatter(text: str) -> dict:
    """The YAML block at the top of a note, or nothing if it will not parse.

    348 of 452 notes here carry `type`, `tags` and `date_created`. That is the
    author's own filing system, and it is worth more than anything inferred.
    Malformed YAML is treated as absent rather than fatal: one bad note must
    not stop the index.
    """
    if not text.strip():
        return {}
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def read_date(fields: dict) -> str:
    """Read the first valid calendar date in note frontmatter."""
    for key in ("date_created", "created", "date"):
        raw = str(fields.get(key, ""))
        found = ISO_DATE.search(raw)
        if found:
            parsed = valid_date(found.group(0))
            if parsed:
                return parsed
        spelled = read_spelled_date(raw)
        if spelled:
            return spelled
    return ""


def read_spelled_date(raw: str) -> str:
    """Dates written out in words, as Obsidian's template produces them.

    `Thursday 26th February 2026 13:38:57` is the shape used by 165 notes in
    this vault. Reading only ISO dates left every one of them undated and so
    invisible to a date filter.

    The month names are matched from a table rather than through strptime,
    because %B follows the machine's locale and this has to give the same
    answer on every device the vault is cloned to.
    """
    match = SPELLED_DATE.search(raw)
    if not match:
        return ""
    day, month_name, year = match.groups()
    month = MONTHS.get(month_name[:3].lower())
    if not month:
        return ""
    return valid_date(f"{year}-{month:02d}-{int(day):02d}")


def valid_date(raw: str) -> str:
    try:
        return date.fromisoformat(raw).isoformat()
    except ValueError:
        return ""


def date_from_name(path: str) -> str:
    """Many notes carry their date in the filename and nowhere else."""
    found = ISO_DATE.search(Path(path).name)
    return valid_date(found.group(0)) if found else ""


def find_links(body: str) -> list[str]:
    """Wikilink targets, once each, ignoring anything inside a fenced code block.

    A documented `[[placeholder]]` in a snippet is not a link between notes, and
    letting it through puts false edges in the graph. Repeats within one note
    collapse to one, so a link count means "notes that link here".
    """
    without_code = FENCED_CODE.sub("", body)
    targets = (target.strip() for target in WIKILINK.findall(without_code))
    return list(dict.fromkeys(target for target in targets if target))


def chunk_note(note: Note, largest: int = 2000, smallest: int = 250) -> list[Chunk]:
    """Keep each passage a contiguous slice of the original note body."""
    sections = merge_short_sections(split_on_headings(note.body), smallest, largest)
    newlines = [match.start() for match in re.finditer("\n", note.body)]
    chunks = []
    for heading, start, end in sections:
        for first, last in split_spans(note.body, start, end, largest):
            text = note.body[first:last]
            if has_content(text):
                chunks.append(Chunk(
                    note.path, heading, text,
                    note.body_start_line + bisect_left(newlines, first),
                    note.body_start_line + bisect_left(newlines, last - 1),
                ))
    return chunks


def has_content(text: str) -> bool:
    return any(character.isalnum() for character in text)


def split_on_headings(body: str) -> list[tuple[str, int, int]]:
    """Heading paths and source offsets, with fenced code left intact."""
    headings = []
    ancestors: list[tuple[int, str]] = []
    fence = None
    offset = 0
    for line in body.splitlines(keepends=True):
        text = line.rstrip("\r\n")
        marker = FENCE.match(text)
        if marker:
            marks, trailing = marker.groups()
            if fence is None:
                fence = (marks[0], len(marks))
            elif marks[0] == fence[0] and len(marks) >= fence[1] and not trailing.strip():
                fence = None
        elif fence is None and (heading := HEADING.match(text)):
            level = len(heading.group(1))
            while ancestors and ancestors[-1][0] >= level:
                ancestors.pop()
            ancestors.append((level, heading.group(2).strip()))
            headings.append((offset, offset + len(line), " > ".join(name for _, name in ancestors)))
        offset += len(line)
    if not headings:
        return [("", 0, len(body))] if has_content(body) else []

    sections = []
    if has_content(body[:headings[0][0]]):
        sections.append(("", 0, headings[0][0]))
    for position, (start, content_start, heading) in enumerate(headings):
        end = headings[position + 1][0] if position + 1 < len(headings) else len(body)
        if has_content(body[content_start:end]):
            sections.append((heading, start, end))
    return sections


def merge_short_sections(
    sections: list[tuple[str, int, int]], smallest: int, largest: int
) -> list[tuple[str, int, int]]:
    """Merge neighbouring spans without removing their headings or separators."""
    merged = []
    for heading, start, end in sections:
        if not merged:
            merged.append((heading, start, end))
            continue
        previous_heading, previous_start, previous_end = merged[-1]
        either_is_short = previous_end - previous_start < smallest or end - start < smallest
        if either_is_short and end - previous_start <= largest:
            merged[-1] = (shared_heading(previous_heading, heading), previous_start, end)
            continue
        merged.append((heading, start, end))
    return merged


def shared_heading(left: str, right: str) -> str:
    """The deepest heading that still describes both merged sections.

    Keeping the first section's heading would label half the chunk with a
    heading that does not cover it.
    """
    shared = []
    for left_part, right_part in zip(left.split(" > "), right.split(" > ")):
        if left_part != right_part:
            break
        shared.append(left_part)
    return " > ".join(shared)


def split_spans(body: str, start: int, end: int, largest: int) -> list[tuple[int, int]]:
    """Prefer paragraph and line boundaries, falling back to a character limit."""
    spans = []
    while end - start > largest:
        cut = start + largest
        paragraph = body.rfind("\n\n", start + 1, cut)
        newline = body.rfind("\n", start + 1, cut)
        if paragraph != -1 and has_content(body[start:paragraph + 2]):
            cut = paragraph + 2
        elif newline != -1 and has_content(body[start:newline + 1]):
            cut = newline + 1
        spans.append((start, cut))
        start = cut
    if start < end:
        spans.append((start, end))
    return spans
