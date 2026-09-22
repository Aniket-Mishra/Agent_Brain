"""The graph tab: what sits near a note in the vault's own link structure."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import streamlit as st

from brain import index, links


def render(connection: sqlite3.Connection) -> None:
    paths = index.all_paths(connection)
    if not paths:
        st.info("Nothing indexed yet.")
        return

    ordered = most_linked_first(connection, paths)
    chosen = st.session_state.get("selected_note")
    start = ordered.index(chosen) if chosen in ordered else 0

    controls = st.columns([3, 1])[0]
    with controls:
        focus = st.selectbox(
            "Centre on", ordered, index=start, format_func=lambda path: Path(path).stem
        )
        hops = st.slider("Hops", 1, 3, 1)

    nodes, edges = links.neighbourhood(connection, focus, hops)
    if len(nodes) <= 1:
        st.info(f"{Path(focus).stem} has no links in or out.")
        return

    st.caption(f"{len(nodes)} notes, {len(edges)} links")
    st.graphviz_chart(links.to_dot(nodes, edges, focus), use_container_width=True)
    unlinked = len(links.orphans(connection))
    if unlinked:
        st.caption(f"{unlinked} notes in the vault have no links in or out.")

    if st.button(f"Open {Path(focus).stem} in Browse"):
        st.session_state["selected_note"] = focus


def most_linked_first(connection: sqlite3.Connection, paths: list[str]) -> list[str]:
    """Busiest notes at the top of the picker, since those are worth looking at."""
    linked = [path for path, _count in links.most_linked(connection, limit=len(paths))]
    already = set(linked)
    return linked + [path for path in paths if path not in already]
