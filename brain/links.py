"""The vault's own link graph: what points where, and what sits nearby."""

from __future__ import annotations

import sqlite3
from pathlib import Path

# A map-of-content note links to everything, so expanding through one drags in
# most of the vault and the picture stops meaning anything.
BUSY_NOTE_LINKS = 25


def outgoing(connection: sqlite3.Connection, path: str) -> list[str]:
    return [
        row["target"]
        for row in connection.execute(
            "SELECT DISTINCT target FROM links WHERE source = ? ORDER BY target", (path,)
        )
    ]


def backlinks(connection: sqlite3.Connection, path: str) -> list[str]:
    return [
        row["source"]
        for row in connection.execute(
            "SELECT DISTINCT source FROM links WHERE target = ? ORDER BY source", (path,)
        )
    ]


def neighbours(connection: sqlite3.Connection, path: str) -> list[str]:
    """Notes one hop away, in either direction."""
    return sorted(set(outgoing(connection, path)) | set(backlinks(connection, path)))


def degree(connection: sqlite3.Connection, path: str) -> int:
    return len(set(outgoing(connection, path)) | set(backlinks(connection, path)))


def neighbourhood(
    connection: sqlite3.Connection, path: str, hops: int = 1
) -> tuple[set[str], set[tuple[str, str]]]:
    """Notes within `hops` of the focus, and the edges between them."""
    reached = {path}
    frontier = {path}
    for hop in range(hops):
        nearby = expand(connection, frontier, unrestricted=hop == 0)
        frontier = nearby - reached
        reached |= nearby
        if not frontier:
            break

    placeholders = ",".join("?" * len(reached))
    edges = {
        (row["source"], row["target"])
        for row in connection.execute(
            f"SELECT source, target FROM links "
            f"WHERE source IN ({placeholders}) AND target IN ({placeholders})",
            [*reached, *reached],
        )
    }
    return reached, edges


def expand(connection: sqlite3.Connection, frontier: set[str], unrestricted: bool) -> set[str]:
    """Everything one hop from the frontier.

    The focus always expands, because someone who centred on their busiest note
    wants to see around it. The cap only stops later hops routing through some
    other hub and dragging in the whole vault.
    """
    nearby: set[str] = set()
    for note in frontier:
        if unrestricted or degree(connection, note) <= BUSY_NOTE_LINKS:
            nearby |= set(neighbours(connection, note))
    return nearby


def to_dot(nodes: set[str], edges: set[tuple[str, str]], focus: str) -> str:
    """Render the neighbourhood as Graphviz source.

    Streamlit draws a DOT string in the browser, so this needs no graph
    library and no Graphviz binary.
    """
    lines = [
        "digraph vault {",
        "  rankdir=LR;",
        "  bgcolor=transparent;",
        '  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=10 '
        'color="#cccccc" fillcolor="#f4f4f6"];',
        '  edge [color="#999999" arrowsize=0.6];',
    ]
    for note in sorted(nodes):
        label = escape(Path(note).stem)
        highlight = ' fillcolor="#ffd9d0" color="#ff4b4b"' if note == focus else ""
        lines.append(f'  "{escape(note)}" [label="{label}"{highlight}];')
    for source, target in sorted(edges):
        lines.append(f'  "{escape(source)}" -> "{escape(target)}";')
    lines.append("}")
    return "\n".join(lines)


def escape(text: str) -> str:
    return text.replace("\\", "").replace('"', "'")


def most_linked(connection: sqlite3.Connection, limit: int = 20) -> list[tuple[str, int]]:
    rows = connection.execute(
        "SELECT target, COUNT(*) AS total FROM links GROUP BY target "
        "ORDER BY total DESC, target LIMIT ?",
        (limit,),
    ).fetchall()
    return [(row["target"], row["total"]) for row in rows]


def orphans(connection: sqlite3.Connection) -> list[str]:
    """Notes nothing links to and which link nowhere."""
    rows = connection.execute(
        "SELECT path FROM notes WHERE path NOT IN (SELECT source FROM links) "
        "AND path NOT IN (SELECT target FROM links) ORDER BY path"
    ).fetchall()
    return [row["path"] for row in rows]
