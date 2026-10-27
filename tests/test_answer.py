from urllib.parse import parse_qs, urlsplit

import pytest

from brain import answer, loop, ollama, tools
from brain.search import Hit, Scope
from views.chat import note_uri


def test_prompt_admits_raw_prefix_with_accurate_lines() -> None:
    hit = Hit(1, "Repeated.md", "Repeated", "", "αβ\n" * 500, start_line=10, end_line=509)
    messages, evidence = answer.build_messages("Read it", [hit], budget=1000)
    assert len(evidence) == 1 and evidence[0].truncated
    retained = evidence[0]
    assert hit.text.startswith(retained.text)
    assert retained.end_line == 10 + retained.text[:-1].count("\n")
    assert retained.location in messages[-1]["content"]
    assert retained.excerpt in messages[-1]["content"]
    assert loop.request_size(messages, None) <= 1000


@pytest.mark.parametrize("content,status,done_reason", [
    ("The owner is Rhea [E1].", "answered", "stop"),
    ("The owner is Rhea [E99].", "insufficient_evidence", "stop"),
    ("The owner is Rhea.", "insufficient_evidence", "stop"),
    ("Insufficient evidence: no owner is recorded.", "insufficient_evidence", "stop"),
    ("Partial answer [E1].", "limit", "length"),
])
def test_fixed_search_uses_shared_citation_checks(monkeypatch, content, status, done_reason) -> None:
    hit = Hit(1, "Owner.md", "Owner", "Decision", "The owner is Rhea.", start_line=4, end_line=4)
    monkeypatch.setattr(tools, "search", lambda *_args: tools.ToolResult("", (hit,)))
    requests = []

    def complete(messages, *_args, **_options):
        requests.append(messages)
        return ollama.Reply(content, done_reason=done_reason)

    monkeypatch.setattr(ollama, "complete", complete)
    turn = answer.answer(None, "Who owns it?", "local", "embed", 8, 8192, Scope())
    assert turn.status == status and turn.model_calls == len(requests) == 1
    assert turn.evidence == (hit,)
    assert hit.location in requests[0][-1]["content"]


def test_oversized_question_stops_before_search(monkeypatch) -> None:
    monkeypatch.setattr(tools, "search", lambda *_args: pytest.fail("Oversized prompt reached retrieval"))
    turn = answer.answer(None, "α" * 5000, "local", "embed", 8, 2048, Scope())
    assert turn.status == "limit" and not turn.evidence and not turn.model_calls


def test_omitted_passages_never_become_displayed_evidence(monkeypatch) -> None:
    hits = (
        Hit(1, "First.md", "First", "", "First fact.\n" * 1000, start_line=1, end_line=1000),
        Hit(2, "Second.md", "Second", "", "Never admitted.", start_line=1, end_line=1),
    )
    monkeypatch.setattr(tools, "search", lambda *_args: tools.ToolResult("", hits))
    monkeypatch.setattr(ollama, "complete", lambda *_args, **_options: ollama.Reply("First fact [E1]."))
    turn = answer.answer(None, "Fact?", "local", "embed", 8, 2048, Scope())
    assert turn.paths == ("First.md",)
    assert turn.evidence[0].truncated
    assert "Never admitted" not in turn.evidence[0].text


def test_obsidian_link_encodes_exact_note_and_rejects_escape(tmp_path) -> None:
    vault = tmp_path / "Notes vault"
    vault.mkdir()
    note = vault / "Δ decision #1.md"
    note.write_text("Fact")
    uri = note_uri(str(vault), note.name)
    assert uri.startswith("obsidian://open?")
    assert "%20" in uri and "+" not in uri
    assert parse_qs(urlsplit(uri).query) == {"path": [str(note)]}
    alias = vault / "Alias.md"
    alias.symlink_to(note)
    assert parse_qs(urlsplit(note_uri(str(vault), alias.name)).query) == {"path": [str(alias)]}
    outside = tmp_path / "outside.md"
    outside.write_text("Outside")
    (vault / "escape.md").symlink_to(outside)
    assert note_uri(str(vault), "escape.md") is None
    assert note_uri(str(vault), "../outside.md") is None


def test_repeated_chunk_keeps_the_admitted_source_span() -> None:
    hit = Hit(1, "Long.md", "Long", "", "Long fact.\n" * 100, start_line=1, end_line=100)
    evidence = []
    loop.admit_result(tools.ToolResult("", (hit,)), 250, evidence, [], "read")
    assert evidence[0].truncated and evidence[0].end_line < hit.end_line
    repeated = loop.admit_result(tools.ToolResult("", (hit,)), 250, evidence, [], "read")
    assert repeated == f"[E1] already retrieved: {evidence[0].location}"
