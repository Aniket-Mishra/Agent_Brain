"""Grouping the vault into themes, for questions no single passage answers.

"What are the main themes in my thesis notes" cannot be answered by retrieving
eight passages, because the answer is a property of the whole set. Clustering
the vectors already in the index gives the groups, and the model is asked to
name each one. That is eight calls rather than one per note.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brain import ollama
from brain.search import Scope, note_vectors

SMALLEST_THEME = 3

NAMING_PROMPT = """These note titles were grouped together because they are about similar things.

{titles}

Reply with one short phrase, at most six words, naming what they have in common.
No preamble, no quotes, no full stop. Just the phrase."""


@dataclass
class Theme:
    name: str
    paths: list[str]

    @property
    def size(self) -> int:
        return len(self.paths)


def find_themes(
    connection: sqlite3.Connection,
    model: str,
    count: int = 8,
    scope: Scope | None = None,
) -> list[Theme]:
    """Cluster the notes in scope, then ask the model to name each cluster."""
    paths, matrix = scoped_vectors(connection, scope or Scope())
    if len(paths) < count:
        return []

    labels = cluster(matrix, count)
    titles = note_titles(connection, paths)

    found = []
    for group in range(count):
        members = [path for path, label in zip(paths, labels) if label == group]
        # A pair of notes is a coincidence, not a theme.
        if len(members) < SMALLEST_THEME:
            continue
        found.append(Theme(name=name_group(members, titles, model), paths=members))
    return make_names_distinct(sorted(found, key=lambda theme: -theme.size))


def make_names_distinct(found: list[Theme]) -> list[Theme]:
    """Separate themes the model happened to give the same name.

    Two clusters of daily notes are genuinely different groups, and showing
    "Daily Notes" twice tells the reader nothing about which is which. The
    folder they mostly live in is the cheapest true distinguisher.
    """
    seen: set[str] = set()
    distinct = []
    for theme in found:
        name = theme.name
        if name in seen:
            name = f"{theme.name} · {common_folder(theme.paths)}"
        seen.add(name)
        distinct.append(Theme(name=name, paths=theme.paths))
    return distinct


def scoped_vectors(
    connection: sqlite3.Connection, scope: Scope
) -> tuple[list[str], np.ndarray]:
    paths, matrix = note_vectors(connection)
    if scope.is_everything or not paths:
        return paths, matrix

    where, values = scope.clause()
    allowed = {
        row["path"]
        for row in connection.execute(f"SELECT path FROM notes WHERE {where}", values)
    }
    keep = [position for position, path in enumerate(paths) if path in allowed]
    if not keep:
        return [], np.zeros((0, 0), dtype=np.float32)
    return [paths[position] for position in keep], matrix[keep]


def cluster(matrix: np.ndarray, count: int, rounds: int = 25, seed: int = 7) -> np.ndarray:
    """k-means over unit vectors, which is about twenty lines of numpy.

    scikit-learn would do this better and weighs 30 MB. At 452 notes the
    difference is invisible, so the dependency does not earn its place.
    """
    generator = np.random.default_rng(seed)
    centres = matrix[generator.choice(len(matrix), size=count, replace=False)].copy()

    labels = np.zeros(len(matrix), dtype=np.int64)
    for _round in range(rounds):
        labels = np.argmax(matrix @ centres.T, axis=1)
        if not recentre(matrix, labels, centres):
            break
    return labels


def recentre(matrix: np.ndarray, labels: np.ndarray, centres: np.ndarray) -> bool:
    """Move each centre onto the mean of its members. True if any of them moved."""
    moved = False
    for group in range(len(centres)):
        members = matrix[labels == group]
        if not len(members):
            continue
        centre = unit(members.mean(axis=0))
        if np.allclose(centre, centres[group]):
            continue
        centres[group] = centre
        moved = True
    return moved


def unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    return vector if length == 0 else vector / length


def note_titles(connection: sqlite3.Connection, paths: list[str]) -> dict[str, str]:
    rows = connection.execute("SELECT path, title FROM notes").fetchall()
    known = {row["path"]: row["title"] for row in rows}
    return {path: known.get(path, Path(path).stem) for path in paths}


def name_group(members: list[str], titles: dict[str, str], model: str) -> str:
    """Name a cluster from its titles, falling back to the commonest folder."""
    sample = [titles.get(path, Path(path).stem) for path in members[:20]]
    try:
        reply = "".join(
            ollama.stream_reply(
                [{"role": "user", "content": NAMING_PROMPT.format(titles="\n".join(sample))}],
                model,
                context_size=4096,
            )
        )
    except ollama.OllamaError:
        return common_folder(members)
    name = reply.strip().strip('"').splitlines()[0] if reply.strip() else ""
    return name[:60] or common_folder(members)


def common_folder(members: list[str]) -> str:
    folders = [str(Path(path).parent) for path in members]
    return max(set(folders), key=folders.count) if folders else "unnamed"
