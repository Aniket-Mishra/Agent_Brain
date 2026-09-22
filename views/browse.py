"""The browse tab: find a note, read it, see what links to it."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import streamlit as st

from brain import index, links


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
    """Filter across everything, or drill into one folder.

    A tree of all 41 folders is a longer list than the notes inside most of
    them, so the folder is a single picker and only its notes get rendered.
    """
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
    # A fixed height keeps the list scrolling on its own. Without it a folder of
    # 174 notes pushes the note being read off the bottom of the page.
    with st.container(height=520):
        for path in paths:
            if st.button(Path(path).stem, key=f"{key}:{path}", use_container_width=True):
                st.session_state["selected_note"] = path
    return st.session_state.get("selected_note")


def show_note(connection: sqlite3.Connection, vault: Path, path: str | None) -> None:
    if not path:
        st.info("Pick a note on the left.")
        return

    st.subheader(Path(path).stem)
    st.caption(path)

    pointing_here = links.backlinks(connection, path)
    pointing_out = links.outgoing(connection, path)
    if pointing_here or pointing_out:
        left, right = st.columns(2)
        left.caption(f"**{count_notes(pointing_here)} in**")
        for source in pointing_here[:15]:
            left.markdown(f"- {Path(source).stem}")
        right.caption(f"**{count_notes(pointing_out)} out**")
        for target in pointing_out[:15]:
            right.markdown(f"- {Path(target).stem}")

    full_path = vault / path
    if not full_path.is_file():
        return
    st.divider()
    with st.container(height=520):
        st.markdown(full_path.read_text(encoding="utf-8", errors="replace"))


def count_notes(paths: list[str]) -> str:
    return "1 note" if len(paths) == 1 else f"{len(paths)} notes"
