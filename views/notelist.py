"""One clickable list of notes, shared by every tab that shows one."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import streamlit as st


def labels_for(paths: list[str]) -> dict[str, str]:
    """Add folder labels only when note names collide."""
    stems = Counter(Path(path).stem for path in paths)
    return {
        path: (
            Path(path).stem
            if stems[Path(path).stem] == 1
            else f"{Path(path).stem}  ·  {Path(path).parent.name or 'top level'}"
        )
        for path in paths
    }


def show(paths: list[str], key: str, shown: int = 40) -> None:
    """Render the list. Clicking a note hands it to the Browse tab."""
    labels = labels_for(paths[:shown])
    for path in paths[:shown]:
        if st.button(labels[path], key=f"{key}:{path}", use_container_width=True):
            st.session_state["selected_note"] = path
    if len(paths) > shown:
        st.caption(f"and {len(paths) - shown} more")
