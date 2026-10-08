"""Fixed search chat, with an opt-in loop for exploring indexed notes."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote, urlencode

import streamlit as st

from brain import answer, loop, ollama
from brain.search import Hit, Scope


def render(
    connection: sqlite3.Connection,
    chat_model: str,
    embedding_model: str,
    passages: int,
    context_size: int,
    num_predict: int,
    scope: Scope,
    vault_identity: str,
) -> None:
    history = st.session_state.setdefault("history", [])
    context = (vault_identity, scope)
    if st.session_state.get("chat_context") != context:
        st.session_state.pop("chat_prior", None)
        st.session_state["chat_context"] = context
    experimental = st.checkbox(
        "Explore notes with tools (experimental)", key="explore_with_tools",
        help=(
            "Allows extra searches and link navigation. "
            "Fixed search chat remains the default while answer quality is evaluated."
        ),
    )

    for item in history:
        with st.chat_message(item["role"]):
            if item.get("turn") is not None:
                show_turn(item["turn"], item["context"][0])
            else:
                st.markdown(item["text"])

    if not scope.is_everything:
        st.caption("Answering from the filtered notes only. Clear the filters to widen it.")

    question = st.chat_input("Ask your vault something", submit_mode="disable")
    if not question:
        return

    history.append({"role": "user", "text": question, "context": context})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = None
        try:
            if experimental:
                status = st.status("Exploring notes", expanded=False)
                result = loop.answer(
                    connection, question, prior=st.session_state.get("chat_prior"),
                    chat_model=chat_model, embed_model=embedding_model,
                    passages=passages, context_size=context_size, scope=scope,
                    limits=loop.Limits(num_predict=num_predict),
                    prior_scope=scope,
                    progress=lambda name: status.update(label=f"Exploring notes: {name}"),
                )
                status.update(
                    label="Exploration complete" if result.status != "limit" else "Exploration stopped at a limit",
                    state="complete" if result.status != "limit" else "error",
                )
            else:
                with st.spinner("Searching your notes"):
                    result = answer.answer(
                        connection, question, chat_model, embedding_model, passages,
                        context_size, scope, num_predict=num_predict,
                    )
            show_turn(result, vault_identity)
        except (ollama.OllamaError, OSError, sqlite3.Error, ValueError) as error:
            if status is not None:
                status.update(label="Exploration failed", state="error")
            st.error(str(error))
            return

    history.append({
        "role": "assistant", "text": result.text, "turn": result,
        "context": context,
    })
    if result.status == "answered":
        st.session_state["chat_prior"] = (question, result.text)


def show_turn(turn: loop.Turn, vault_identity: str) -> None:
    st.markdown(turn.text)
    if turn.status == "limit":
        st.caption("Incomplete result")
    show_sources(turn.evidence, turn.cited, vault_identity)
    if turn.facts:
        with st.expander(f"{len(turn.facts)} computed index results"):
            for number, fact in enumerate(turn.facts, 1):
                state = "cited" if number in turn.cited_facts else "retrieved"
                st.markdown(f"**[I{number}] {state}**")
                st.text(fact.text)


def note_uri(vault_identity: str, path: str) -> str | None:
    vault = Path(vault_identity).resolve()
    note = vault / path
    if not note.resolve().is_relative_to(vault) or not note.is_file():
        return None
    return "obsidian://open?" + urlencode({"path": str(note)}, quote_via=quote)


def show_sources(hits: Sequence[Hit], cited: tuple[int, ...], vault_identity: str) -> None:
    if not hits:
        return
    with st.expander(f"{len(hits)} retrieved passages"):
        st.caption("Source lines refer to the indexed note. Later edits can move them.")
        for number, hit in enumerate(hits, start=1):
            identifier = f"[E{number}]"
            state = " · cited" if number in cited else " · retrieved"
            st.markdown(f"**{identifier} {hit.title}{state}**")
            details = [hit.location]
            if hit.heading:
                details.append(hit.heading)
            details.extend(part for part in (hit.kind, hit.created) if part)
            if hit.found_by:
                details.append(f"found by {', '.join(sorted(hit.found_by))}")
            st.caption("  ·  ".join(details))
            st.text(hit.excerpt)
            uri = note_uri(vault_identity, hit.path)
            if uri:
                st.link_button("Open note in Obsidian", uri)
