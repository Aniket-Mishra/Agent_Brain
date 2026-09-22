"""The chat tab: ask a question, stream the answer, show what it came from."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import streamlit as st

from brain import ollama
from brain.answer import NOTHING_FOUND, build_messages, source_budget
from brain.search import Hit, Scope, search


def render(
    connection: sqlite3.Connection,
    chat_model: str,
    embedding_model: str,
    passages: int,
    context_size: int,
    scope: Scope,
) -> None:
    history = st.session_state.setdefault("history", [])

    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["text"])
            if turn.get("sources"):
                show_sources(turn["sources"])

    if not scope.is_everything:
        st.caption("Answering from the filtered notes only. Clear the filters to widen it.")

    question = st.chat_input("Ask your vault something")
    if not question:
        return

    history.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your notes"):
                hits = search(
                    connection, question, embedding_model, limit=passages, scope=scope
                )
            if not hits:
                st.markdown(NOTHING_FOUND)
                history.append({"role": "assistant", "text": NOTHING_FOUND})
                return
            messages = build_messages(question, hits, source_budget(context_size))
            answer = st.write_stream(ollama.stream_reply(messages, chat_model, context_size))
        except ollama.OllamaError as error:
            st.error(str(error))
            return
        show_sources(hits)

    history.append({"role": "assistant", "text": answer, "sources": hits})


def show_sources(hits: list[Hit]) -> None:
    with st.expander(f"{len(hits)} sources"):
        for number, hit in enumerate(hits, start=1):
            label = Path(hit.path).stem
            if hit.heading:
                label += f"  ·  {hit.heading}"
            st.markdown(f"**[{number}] {label}**")
            details = [hit.path, f"found by {', '.join(sorted(hit.found_by))}"]
            if hit.kind:
                details.insert(1, hit.kind)
            if hit.created:
                details.insert(1, hit.created)
            st.caption("  ·  ".join(details))
            st.text(hit.text[:1500])
