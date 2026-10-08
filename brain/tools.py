"""Read-only tools over the indexed vault, shared by the CLI and model loop."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace
from pathlib import PureWindowsPath

from brain import search as retrieval
from brain.search import Hit, Scope

QUERY_BYTES = 8192
PATH_BYTES = 4096
READ_CHARACTERS = 6000
TRUNCATED = "\n[truncated]"


@dataclass(frozen=True)
class ToolResult:
    text: str
    evidence: tuple[Hit, ...] = ()
    paths: tuple[str, ...] = ()


def validate_query(query: str) -> None:
    try:
        valid = (
            isinstance(query, str)
            and bool(query.strip())
            and len(query.encode("utf-8")) <= QUERY_BYTES
        )
    except UnicodeEncodeError:
        valid = False
    if not valid:
        raise ValueError("query must be nonempty text up to 8192 UTF-8 bytes")


def _readable(connection: sqlite3.Connection, path: str, scope: Scope | None) -> bool:
    if not isinstance(path, str):
        return False
    try:
        valid = (
            len(path.encode("utf-8")) <= PATH_BYTES
            and "\\" not in path
            and not PureWindowsPath(path).drive
            and all(part not in ("", ".", "..") for part in path.split("/"))
        )
    except UnicodeEncodeError:
        return False
    if not valid:
        return False
    where, values = (scope or Scope()).clause()
    return connection.execute(
        f"SELECT 1 FROM notes WHERE notes.path = ? AND {where}", [path, *values]
    ).fetchone() is not None


def search(
    connection: sqlite3.Connection,
    query: str,
    embed_model: str,
    limit: int = 8,
    scope: Scope | None = None,
) -> ToolResult:
    validate_query(query)
    if type(limit) is not int or not 1 <= limit <= 25:
        raise ValueError("limit must be an integer from 1 to 25")
    hits = retrieval.search(connection, query, embed_model, limit, scope)
    return ToolResult("Indexed passages" if hits else "No matching indexed passages.", tuple(hits))


def read(connection: sqlite3.Connection, path: str, scope: Scope | None = None) -> ToolResult:
    if not _readable(connection, path, scope):
        return ToolResult(f"unreadable: {path}")
    ids = [
        row["id"]
        for row in connection.execute("SELECT id FROM chunks WHERE path = ? ORDER BY id", (path,))
    ]
    hits = retrieval.load_hits(connection, ids, "read")
    total = sum(len(hit.text) for hit in hits)
    remaining = READ_CHARACTERS - (len(TRUNCATED) if total > READ_CHARACTERS else 0)
    kept = []
    for hit in hits:
        if remaining <= 0:
            break
        text = hit.text[:remaining]
        kept.append(replace(hit, text=text))
        remaining -= len(text)
    if total > READ_CHARACTERS:
        kept[-1] = replace(kept[-1], text=kept[-1].text + TRUNCATED)
    label = "Indexed passages" if kept else f"No indexed passages in {path}."
    return ToolResult(label, tuple(kept))


def backlinks(connection: sqlite3.Connection, path: str, scope: Scope | None = None) -> ToolResult:
    if not _readable(connection, path, scope):
        return ToolResult(f"unreadable: {path}")
    where, values = (scope or Scope()).clause()
    rows = connection.execute(
        f"SELECT path, COUNT(*) OVER () AS total FROM notes WHERE {where} "
        "AND path IN (SELECT source FROM links WHERE target = ?) ORDER BY path LIMIT 25",
        [*values, path],
    ).fetchall()
    paths = tuple(row["path"] for row in rows)
    text = f"Backlinks to {path}:\n" + ("\n".join(paths) if paths else "none")
    if rows and rows[0]["total"] > len(rows):
        text += f"\nand {rows[0]['total'] - len(rows)} more"
    return ToolResult(text, paths=paths)


def related(connection: sqlite3.Connection, path: str, scope: Scope | None = None) -> ToolResult:
    if not _readable(connection, path, scope):
        return ToolResult(f"unreadable: {path}")
    found = retrieval.similar_notes(connection, path, limit=5, unlinked_only=True, scope=scope)
    text = f"Related unlinked notes for {path}:\n"
    text += "\n".join(f"{other} (similarity {score:.3f})" for other, score in found) or "none"
    return ToolResult(text, paths=tuple(other for other, _score in found))


def survey(connection: sqlite3.Connection, scope: Scope | None = None) -> ToolResult:
    scope = scope or Scope()
    where, values = scope.clause()
    notes = connection.execute(f"SELECT COUNT(*) FROM notes WHERE {where}", values).fetchone()[0]
    chunks = connection.execute(
        f"SELECT COUNT(*) FROM chunks JOIN notes ON notes.path = chunks.path WHERE {where}", values
    ).fetchone()[0]
    links = connection.execute(
        f"SELECT COUNT(*) FROM links JOIN notes ON notes.path = links.source WHERE {where}", values
    ).fetchone()[0]
    kinds = connection.execute(
        f"SELECT kind, COUNT(*) AS total FROM notes WHERE {where} AND kind != '' "
        "GROUP BY kind ORDER BY total DESC, kind",
        values,
    ).fetchall()
    missing = connection.execute(
        "SELECT raw, COUNT(DISTINCT source) AS total FROM links "
        f"JOIN notes ON notes.path = links.source WHERE {where} AND target IS NULL "
        "GROUP BY raw HAVING total >= 2 ORDER BY total DESC, raw LIMIT 10",
        values,
    ).fetchall()
    stubs = connection.execute(
        f"SELECT path, size, COUNT(*) OVER () AS total FROM notes WHERE {where} AND size <= 200 "
        "ORDER BY size, path LIMIT 10",
        values,
    ).fetchall()
    orphans = connection.execute(
        f"SELECT path, COUNT(*) OVER () AS total FROM notes WHERE {where} AND NOT EXISTS "
        "(SELECT 1 FROM links WHERE target IS NOT NULL "
        "AND (source = notes.path OR target = notes.path)) ORDER BY path LIMIT 10",
        values,
    ).fetchall()
    lines = [
        f"Index facts for scope: kinds={scope.kinds or 'all'}, "
        f"since={scope.since or 'any'}, until={scope.until or 'any'}."
    ]
    lines.extend((f"Notes: {notes}", f"Passages: {chunks}", f"Links from scoped notes: {links}"))
    lines.append("Kinds: " + (", ".join(f"{row['kind']} ({row['total']})" for row in kinds) or "none"))
    lines.append("Missing targets linked by at least two scoped notes:")
    lines.extend(f"{row['raw']} ({row['total']} notes)" for row in missing)
    if not missing:
        lines.append("none")
    lines.append("Stubs (at most 200 characters):")
    lines.extend(f"{row['path']} ({row['size']} characters)" for row in stubs)
    if not stubs:
        lines.append("none")
    elif stubs[0]["total"] > len(stubs):
        lines.append(f"and {stubs[0]['total'] - len(stubs)} more")
    lines.append("Orphans (no resolved links anywhere in the vault):")
    lines.extend(row["path"] for row in orphans)
    if not orphans:
        lines.append("none")
    elif orphans[0]["total"] > len(orphans):
        lines.append(f"and {orphans[0]['total'] - len(orphans)} more")
    paths = tuple(dict.fromkeys(row["path"] for row in [*stubs, *orphans]))
    return ToolResult("\n".join(lines), paths=paths)


def format_result(result: ToolResult) -> str:
    blocks = [result.text] if result.text else []
    for number, hit in enumerate(result.evidence, start=1):
        label = f"[{number}] {hit.path}\nTitle: {hit.title}"
        if hit.heading:
            label += f"\nHeading: {hit.heading}"
        blocks.append(f"{label}\n\n{hit.text}")
    return "\n\n".join(blocks)
