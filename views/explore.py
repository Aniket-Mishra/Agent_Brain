"""The explore tab: what the vault is about, and what it is missing."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import streamlit as st

from brain import links, themes
from views import notelist
from brain.search import Scope


def render(connection: sqlite3.Connection, chat_model: str, scope: Scope) -> None:
    show_themes(connection, chat_model, scope)
    st.divider()
    show_gaps(connection)


def show_themes(connection: sqlite3.Connection, chat_model: str, scope: Scope) -> None:
    st.subheader("Themes")
    st.caption(
        "Groups the notes by what they are about, then names each group. "
        "Uses the vectors already in the index, and one model call per group."
    )

    count = st.slider("Groups", 3, 12, 8)
    if st.button("Find themes", type="primary"):
        with st.spinner("Grouping notes and naming the groups"):
            st.session_state["themes"] = themes.find_themes(
                connection, chat_model, count=count, scope=scope
            )

    found = st.session_state.get("themes")
    if found is None:
        return
    if not found:
        st.info("Not enough notes in the current filter to group.")
        return

    for theme in found:
        with st.expander(f"{theme.name}  ({theme.size} notes)"):
            notelist.show(theme.paths, key=f"theme:{theme.name}", shown=25)


def show_gaps(connection: sqlite3.Connection) -> None:
    st.subheader("Worth your attention")
    unwritten, stubs, orphans = (
        links.unwritten(connection),
        links.stubs(connection),
        links.orphans(connection),
    )

    missing, empty, alone = st.tabs(
        [
            f"Never written ({len(unwritten)})",
            f"Stubs ({len(stubs)})",
            f"Unlinked ({len(orphans)})",
        ]
    )

    with missing:
        st.caption("Linked from your notes, but no note exists yet.")
        for name, total in unwritten[:40]:
            notes = "1 note" if total == 1 else f"{total} notes"
            st.markdown(f"**{name}** · linked from {notes}")

    with empty:
        st.caption("Started and left nearly empty.")
        notelist.show([path for path, _size in stubs], key="stub")

    with alone:
        st.caption("No links in or out. Nothing points at these and they point nowhere.")
        notelist.show(orphans, key="orphan")

