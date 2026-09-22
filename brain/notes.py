"""Reading the vault's markdown: what a note contains, and how it is split up."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

SKIPPED_DIRECTORIES = {".obsidian", ".trash", ".git", ".github", "node_modules", "__pycache__"}
# Below this many characters of actual words, a piece is punctuation and headings.
LEAST_REAL_CONTENT = 20
# Tried in order when a piece is too large. An empty separator means cut on length.
SEPARATORS = ("\n\n", "\n")

FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
FENCED_CODE = re.compile(r"^[ \t]{0,3}(```|~~~).*?^[ \t]{0,3}\1", re.DOTALL | re.MULTILINE)
WIKILINK = re.compile(r"!?\[\[([^\[\]\n|#]+)")
HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
TITLE_FIELD = re.compile(r"^title:[ \t]*(.+?)[ \t]*$", re.MULTILINE)
MARKUP_ONLY = re.compile(r"[#\-*_>`|~\s]+")


@dataclass(frozen=True)
class Note:
    path: str
    title: str
    body: str
    links: list[str]
    digest: str


@dataclass(frozen=True)
class Chunk:
    path: str
    heading: str
    text: str


def embedding_text(title: str, heading: str, text: str) -> str:
    """Breadcrumbs first, then the passage.

    "use port 5432" is ambiguous on its own. The same text under
    "Postgres > Local setup" is not, and the model can only use what it is shown.
    """
    crumbs = " > ".join(part for part in (title, heading) if part)
    return f"{crumbs}\n\n{text}"


def find_notes(vault: Path) -> list[Path]:
    notes = []
    for directory, subdirectories, filenames in os.walk(vault):
        subdirectories[:] = [
            name
            for name in subdirectories
            if name not in SKIPPED_DIRECTORIES and not name.startswith(".")
        ]
        notes.extend(Path(directory) / name for name in filenames if name.endswith(".md"))
    return sorted(notes)


def read_note(path: Path, vault: Path) -> Note:
    raw = path.read_text(encoding="utf-8", errors="replace")
    frontmatter = FRONTMATTER.match(raw)
    body = raw[frontmatter.end():] if frontmatter else raw

    return Note(
        path=path.relative_to(vault).as_posix(),
        title=read_title(frontmatter.group(1) if frontmatter else "", path.stem),
        body=body,
        links=find_links(body),
        digest=hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest(),
    )


def read_title(frontmatter: str, fallback: str) -> str:
    declared = TITLE_FIELD.search(frontmatter)
    if not declared:
        return fallback
    return declared.group(1).strip().strip("\"'") or fallback


def find_links(body: str) -> list[str]:
    """Wikilink targets, ignoring anything inside a fenced code block.

    A documented `[[placeholder]]` in a snippet is not a link between notes, and
    letting it through puts false edges in the graph.
    """
    without_code = FENCED_CODE.sub("", body)
    targets = (target.strip() for target in WIKILINK.findall(without_code))
    return list(dict.fromkeys(target for target in targets if target))


def chunk_note(note: Note, largest: int = 2000, smallest: int = 250) -> list[Chunk]:
    """Split a note at its headings, keeping the heading path with each piece."""
    sections = merge_short_sections(split_on_headings(note.body), smallest, largest)

    chunks = []
    for heading, text in sections:
        pieces = pack(split_to_fit(text, largest), largest, smallest)
        chunks.extend(Chunk(note.path, heading, piece) for piece in pieces if has_content(piece))
    return chunks


def has_content(text: str) -> bool:
    """Reject pieces that are only markdown punctuation.

    A horizontal rule under its own heading is a real section boundary and an
    empty chunk, and indexing it wastes a row that can still be retrieved.
    """
    return len(MARKUP_ONLY.sub("", text)) >= LEAST_REAL_CONTENT


def split_on_headings(body: str) -> list[tuple[str, str]]:
    """Section bodies, each labelled with its full heading path."""
    headings = list(HEADING.finditer(body))
    if not headings:
        return [("", body.strip())] if body.strip() else []

    sections = []
    preamble = body[: headings[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    ancestors: list[tuple[int, str]] = []
    for position, heading in enumerate(headings):
        level = len(heading.group(1))
        while ancestors and ancestors[-1][0] >= level:
            ancestors.pop()
        ancestors.append((level, heading.group(2).strip()))

        end = headings[position + 1].start() if position + 1 < len(headings) else len(body)
        text = body[heading.end(): end].strip()
        if text:
            sections.append((" > ".join(name for _level, name in ancestors), text))
    return sections


def merge_short_sections(
    sections: list[tuple[str, str]], smallest: int, largest: int
) -> list[tuple[str, str]]:
    """Fold tiny sections into their neighbour.

    A heavily nested note otherwise produces one chunk per `###` holding a
    single sentence, which retrieves badly and reads worse.
    """
    merged: list[tuple[str, str]] = []
    for heading, text in sections:
        if not merged:
            merged.append((heading, text))
            continue

        previous_heading, previous_text = merged[-1]
        either_is_short = len(previous_text) < smallest or len(text) < smallest
        if either_is_short and len(previous_text) + len(text) <= largest:
            merged[-1] = (shared_heading(previous_heading, heading), f"{previous_text}\n\n{text}")
            continue
        merged.append((heading, text))
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


def split_to_fit(text: str, largest: int) -> list[str]:
    """Break text down until every piece fits, coarsest separator first.

    Paragraphs normally. A lecture transcript can run 40,000 characters without
    a single blank line in it, so an oversized paragraph falls back to lines,
    and a single oversized line is cut on length.
    """
    if len(text) <= largest:
        return [text]
    for separator in SEPARATORS:
        parts = text.split(separator)
        if len(parts) > 1:
            return [piece for part in parts for piece in split_to_fit(part, largest)]
    return [text[start: start + largest] for start in range(0, len(text), largest)]


def pack(pieces: list[str], largest: int, smallest: int) -> list[str]:
    """Refill the split pieces up to the budget, so chunks are not needlessly small."""
    packed: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > largest:
            packed.append(current)
            current = piece
            continue
        current = f"{current}\n\n{piece}" if current else piece

    if not current:
        return packed
    # A short tail reads as a fragment, so it rides along with the piece before it.
    if packed and len(current) < smallest:
        packed[-1] = f"{packed[-1]}\n\n{current}"
        return packed
    packed.append(current)
    return packed
