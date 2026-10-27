"""The browse tab: find a note, read it, see what links to it."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import streamlit as st

from brain import index, links
from brain.notes import Note, read_note
from brain.search import similar_notes
from views import notelist


def render(connection: sqlite3.Connection, vault: Path) -> None:
    paths = index.all_paths(connection)
    if not paths:
        st.info("Nothing indexed yet.")
        return

    picker, reader = st.columns([1, 2], gap="medium")
    with picker:
        selected = choose_note(paths)
    with reader:
        show_note(connection, vault, selected)


def choose_note(paths: list[str]) -> str | None:
    """Filter all notes or select one folder."""
    needle = st.text_input("Filter", placeholder="part of a note or folder name").strip().lower()
    if needle:
        matching = [path for path in paths if needle in path.lower()]
        st.caption(f"{len(matching)} of {len(paths)} notes")
        return note_buttons(matching[:80], key="filtered")

    folders = group_by_folder(paths)
    names = list(folders)
    current = folder_of(st.session_state.get("selected_note"))
    folder = st.selectbox(
        "Folder",
        names,
        index=names.index(current) if current in names else 0,
        format_func=lambda name: f"{name}  ({len(folders[name])})",
    )
    return note_buttons(folders[folder], key=folder)


def group_by_folder(paths: list[str]) -> dict[str, list[str]]:
    folders: dict[str, list[str]] = {}
    for path in paths:
        folders.setdefault(folder_of(path), []).append(path)
    return dict(sorted(folders.items()))


def folder_of(path: str | None) -> str:
    if not path:
        return ""
    parent = str(Path(path).parent)
    return "(top level)" if parent == "." else parent


def note_buttons(paths: list[str], key: str) -> str | None:
    # Keep the list scrolling without pushing the reader down.
    with st.container(height=520):
        notelist.show(paths, key=key, shown=len(paths))
    return st.session_state.get("selected_note")


def show_note(connection: sqlite3.Connection, vault: Path, path: str | None) -> None:
    if not path:
        st.info("Pick a note on the left.")
        return

    full_path = vault / path
    note = read_note(full_path, vault) if full_path.is_file() else None

    st.subheader(note.title if note else Path(path).stem)
    st.caption(path)
    if note:
        show_metadata(note)

    pointing_here = links.backlinks(connection, path)
    pointing_out = links.outgoing(connection, path)
    if pointing_here or pointing_out:
        left, right = st.columns(2)
        left.caption(f"**{count_notes(pointing_here)} in**")
        show_link_list(left, pointing_here)
        right.caption(f"**{count_notes(pointing_out)} out**")
        show_link_list(right, pointing_out)

    show_similar(connection, path)

    if note is None:
        return
    st.divider()
    with st.container(height=520):
        st.markdown(note.body)


def show_metadata(note: Note) -> None:
    parts = [part for part in (note.created, note.kind) if part]
    if note.tags:
        parts.append(" ".join(f"`{tag}`" for tag in note.tags[:6]))
    if parts:
        st.caption("  ·  ".join(parts))


def show_similar(connection: sqlite3.Connection, path: str) -> None:
    """Close notes this one does not link to, which are links worth making."""
    nearby = similar_notes(connection, path, limit=5)
    if not nearby:
        return
    st.caption("**Related, and not yet linked**")
    notelist.show([other for other, _score in nearby], key="similar", shown=len(nearby))


def count_notes(paths: list[str]) -> str:
    return "1 note" if len(paths) == 1 else f"{len(paths)} notes"


def show_link_list(column, paths: list[str], shown: int = 15) -> None:
    labels = notelist.labels_for(paths[:shown])
    for path in paths[:shown]:
        column.markdown(f"- {labels[path]}")
