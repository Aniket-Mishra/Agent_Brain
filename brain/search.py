"""Finding passages: vector similarity and word matching, merged by rank."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from brain import ollama

# Words that carry no retrieval signal. FTS5 ranks by term frequency, so a note
# repeating "what" four times otherwise outranks the note that answers the
# question. Kept short on purpose: an aggressive list removes words that mean
# something in technical notes, where "not" and "no" can be the whole point.
STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at by for with about from as is are was were be been
    do does did have has had i me my we our you your he she it its they them their this that
    these those what which who when where why how can could should would will may might must
    there here so such than too very just also much many
    """.split()
)

QUOTED_OR_WORD = re.compile(r'"[^"]+"|\S+')
NON_WORD = re.compile(r"[^\w/-]+", re.UNICODE)


@dataclass
class Hit:
    path: str
    title: str
    heading: str
    text: str
    score: float = 0.0
    found_by: set[str] = field(default_factory=set)


def to_match_query(question: str) -> str | None:
    """Rewrite free text into something FTS5 will accept.

    `*`, `:`, `-`, `^` and quotes are operators in FTS5, so a raw question is a
    syntax error waiting to happen. Every bare word is quoted and the terms are
    joined with OR, which is what makes this a ranking rather than a filter.
    """
    phrases, words = [], []
    for token in QUOTED_OR_WORD.findall(question):
        phrase = unwrap_phrase(token)
        if phrase:
            phrases.append(f'"{phrase}"')
            continue
        words.extend(word for word in NON_WORD.sub(" ", token).split() if word)

    meaningful = [word for word in words if word.lower() not in STOPWORDS]
    # Dropping everything would return nothing, which is worse than being vague.
    kept = meaningful if (meaningful or phrases) else words
    terms = phrases + [f'"{word}"' for word in kept]
    return " OR ".join(terms) if terms else None


def unwrap_phrase(token: str) -> str:
    """The inside of a quoted phrase, or empty when the token is a bare word."""
    if len(token) <= 2 or not (token.startswith('"') and token.endswith('"')):
        return ""
    return token[1:-1].replace('"', "").strip()


def by_words(connection: sqlite3.Connection, question: str, limit: int) -> list[Hit]:
    match = to_match_query(question)
    if not match:
        return []
    rows = connection.execute(
        """
        SELECT chunks.path, chunks.heading, chunks.text, notes.title
        FROM chunks_fts
        JOIN chunks ON chunks.id = chunks_fts.rowid
        JOIN notes ON notes.path = chunks.path
        WHERE chunks_fts MATCH ?
        ORDER BY bm25(chunks_fts, 1.0, 0.5) LIMIT ?
        """,
        (match, limit),
    ).fetchall()
    return [to_hit(row, "words") for row in rows]


def by_meaning(connection: sqlite3.Connection, question: str, model: str, limit: int) -> list[Hit]:
    rows = connection.execute(
        "SELECT chunks.id, chunks.path, chunks.heading, chunks.text, chunks.vector, notes.title "
        "FROM chunks JOIN notes ON notes.path = chunks.path WHERE chunks.vector IS NOT NULL"
    ).fetchall()
    if not rows:
        return []

    question_vector = np.asarray(ollama.embed_query(question, model), dtype=np.float32)
    length = float(np.linalg.norm(question_vector))
    if length:
        question_vector /= length

    matrix = np.frombuffer(b"".join(row["vector"] for row in rows), dtype=np.float32)
    matrix = matrix.reshape(len(rows), -1)
    if matrix.shape[1] != question_vector.shape[0]:
        return []

    # ponytail: brute force over every vector. At 2,600 chunks this is a 8 MB
    # dot product taking a few milliseconds, and it is exact. Revisit above
    # roughly 100,000 chunks, where an approximate index starts to earn its keep.
    scores = matrix @ question_vector
    best = np.argsort(-scores)[:limit]
    return [to_hit(rows[position], "meaning") for position in best]


def search(connection: sqlite3.Connection, question: str, model: str, limit: int = 8) -> list[Hit]:
    """Merge word matching and vector similarity by rank.

    BM25 scores and cosine similarities live on different scales, and
    normalising one onto the other breaks whenever the distribution shifts.
    Rank fusion needs no calibration: a passage scores 1/(60 + rank) in each
    list it appears in, so appearing in both beats topping one.
    """
    pool = max(limit * 5, 30)
    rankings = [by_words(connection, question, pool), by_meaning(connection, question, model, pool)]

    merged: dict[tuple[str, str], Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = (hit.path, hit.text[:80])
            existing = merged.setdefault(key, hit)
            existing.score += 1 / (60 + rank)
            existing.found_by |= hit.found_by

    ordered = sorted(merged.values(), key=lambda hit: -hit.score)
    return limit_per_note(ordered, allowed=2)[:limit]


def limit_per_note(hits: list[Hit], allowed: int) -> list[Hit]:
    """Stop one long note filling the whole answer with itself."""
    seen: dict[str, int] = {}
    kept = []
    for hit in hits:
        if seen.get(hit.path, 0) >= allowed:
            continue
        seen[hit.path] = seen.get(hit.path, 0) + 1
        kept.append(hit)
    return kept


def to_hit(row: sqlite3.Row, found_by: str) -> Hit:
    return Hit(
        path=row["path"],
        title=row["title"],
        heading=row["heading"],
        text=row["text"],
        found_by={found_by},
    )
