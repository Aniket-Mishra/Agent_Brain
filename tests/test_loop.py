import copy
import json
from pathlib import Path

import pytest

from brain import index, loop, ollama, tools
from brain.notes import Note
from brain.search import Hit, Scope


@pytest.fixture
def indexed_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    notes = [
        Note("Alpha.md", "Storage decision", "We chose SQLite to store local passages.", [], "alpha", 300, "research", created="2026-03-10"),
        Note("Beta.md", "Index recovery", "Rebuild the derived index after changing models.", [], "beta", 300, "research", created="2026-03-11"),
        Note("Hidden.md", "Private daily note", "Hidden daily note contains pineapple secret.", [], "hidden", 300, "daily", created="2026-03-12"),
    ]
    connection = index.connect(tmp_path / "loop.db")
    for note in notes:
        index.store_note(connection, note)
        connection.execute("UPDATE chunks SET vector = ? WHERE path = ?", (index.pack([1.0, 0.0]), note.path))
    connection.commit()
    monkeypatch.setattr(ollama, "embed_query", lambda *_args: [1.0, 0.0])

    def no_network(*_args, **_kwargs):
        pytest.fail("offline loop test attempted an Ollama request")

    monkeypatch.setattr(ollama, "request", no_network)
    yield connection
    connection.close()


def scripted(monkeypatch, replies):
    requests = []
    remaining = iter(replies)

    def complete(messages, model, context_size, tools=None, **options):
        requests.append((copy.deepcopy(messages), tools, options))
        return next(remaining)

    monkeypatch.setattr(loop.ollama, "complete", complete)
    return requests


def test_search_then_read_keeps_citations_and_counts_work(indexed_loop, monkeypatch) -> None:
    requests = scripted(monkeypatch, [
        ollama.Reply("", (ollama.ToolCall("search", {"query": "SQLite"}),), 12, 3),
        ollama.Reply("", (ollama.ToolCall("read", {"path": "Alpha.md"}),), 15, 5),
        ollama.Reply("SQLite stores local passages [E1].", prompt_eval_count=20, eval_count=6),
    ])
    turn = loop.answer(indexed_loop, "What stores local passages?", passages=1)
    assert turn.status == "answered"
    assert turn.cited == (1,)
    assert turn.paths == ("Alpha.md",)
    assert len(turn.evidence) == 1
    assert turn.evidence[0].title == "Storage decision"
    assert turn.tool_names == ("search", "read")
    assert (turn.model_calls, turn.prompt_eval_count, turn.eval_count) == (3, 47, 14)
    for messages, _tools, _options in requests[1:]:
        shown = "\n".join(message.get("content", "") for message in messages)
        assert "[E1]" in shown and "Storage decision" in shown
        assert "[E2]" not in shown


def test_reasoning_is_replayed_for_tools_but_not_displayed(indexed_loop, monkeypatch) -> None:
    requests = scripted(monkeypatch, [
        ollama.Reply("", (ollama.ToolCall("read", {"path": "Alpha.md"}),), thinking="Inspect Alpha [E99]"),
        ollama.Reply("SQLite stores passages [E1].", thinking="Ignore this draft [E99]"),
    ])
    turn = loop.answer(indexed_loop, "Which store is used?")
    assert requests[1][0][2]["thinking"] == "Inspect Alpha [E99]"
    assert turn.text == "SQLite stores passages [E1]."
    assert turn.status == "answered" and turn.cited == (1,)


@pytest.mark.parametrize("content", ["SQLite stores passages.", "SQLite stores passages [E99].", "Index count [I99]."])
def test_unproven_answers_are_not_marked_answered(indexed_loop, monkeypatch, content) -> None:
    scripted(monkeypatch, [
        ollama.Reply("", (ollama.ToolCall("read", {"path": "Alpha.md"}),)),
        ollama.Reply(content),
    ])
    turn = loop.answer(indexed_loop, "What stores local passages?")
    assert turn.status == "insufficient_evidence"
    assert 99 not in turn.cited and 99 not in turn.cited_facts


@pytest.mark.parametrize("content", ["SQLite stores passages.", "Insufficient evidence: No note answers this."])
def test_no_retrieved_evidence_cannot_support_an_answer(indexed_loop, monkeypatch, content) -> None:
    scripted(monkeypatch, [ollama.Reply(content)])
    turn = loop.answer(indexed_loop, "What did I decide?")
    assert turn.status == "insufficient_evidence"
    assert not turn.evidence and not turn.facts and not turn.paths


def test_prior_citations_do_not_become_current_evidence(indexed_loop, monkeypatch) -> None:
    scripted(monkeypatch, [ollama.Reply("SQLite stores passages [E1].")])
    turn = loop.answer(indexed_loop, "What stores passages?", prior=("Previous question", "SQLite [E1]."))
    assert turn.status == "insufficient_evidence" and not turn.paths


def test_empty_read_label_is_not_a_citable_index_fact(indexed_loop, monkeypatch) -> None:
    indexed_loop.execute("DELETE FROM chunks WHERE path = 'Alpha.md'")
    scripted(monkeypatch, [
        ollama.Reply("", (ollama.ToolCall("read", {"path": "Alpha.md"}),)),
        ollama.Reply("No indexed passages exist [I1]."),
    ])
    turn = loop.answer(indexed_loop, "Read storage")
    assert turn.status == "insufficient_evidence" and not turn.facts


def test_survey_uses_index_fact_citations(indexed_loop, monkeypatch) -> None:
    scripted(monkeypatch, [
        ollama.Reply("", (ollama.ToolCall("survey", {}),)),
        ollama.Reply("Two research notes are indexed [I1]."),
    ])
    turn = loop.answer(indexed_loop, "How many research notes exist?", scope=Scope(kinds=("research",)))
    assert turn.status == "answered"
    assert turn.cited_facts == (1,) and not turn.cited
    assert len(turn.facts) == 1 and not turn.evidence
    assert "Notes: 2" in turn.facts[0].text
    assert set(turn.paths) == {"Alpha.md", "Beta.md"}


def test_invalid_survey_cannot_be_cited_as_index_fact(indexed_loop, monkeypatch) -> None:
    requests = scripted(monkeypatch, [
        ollama.Reply("", (ollama.ToolCall("survey", {"scope": "all"}),)),
        ollama.Reply("No notes exist [I1]."),
    ])
    turn = loop.answer(indexed_loop, "How many notes exist?")
    assert turn.status == "insufficient_evidence" and not turn.facts
    assert not turn.cited_facts and not turn.paths
    result = requests[-1][0][-1]["content"]
    assert result.startswith("error:") and "[I1]" not in result


def test_invalid_tool_calls_and_scope_bypass_never_admit_hidden_notes(indexed_loop, monkeypatch) -> None:
    requests = scripted(monkeypatch, [
        ollama.Reply("", (
            ollama.ToolCall("delete_note", {"path": "Alpha.md"}),
            ollama.ToolCall("search", {"query": 7}),
            ollama.ToolCall("read", {}),
            ollama.ToolCall("read", {"path": "Hidden.md"}),
            ollama.ToolCall("survey", {"scope": "all"}),
        )),
        ollama.Reply("Insufficient evidence: Tool requests failed."),
    ])
    turn = loop.answer(indexed_loop, "Read notes", scope=Scope(kinds=("research",)))
    assert turn.status == "insufficient_evidence"
    assert "Hidden.md" not in turn.paths
    shown = [message["content"] for message in requests[-1][0] if message["role"] == "tool"]
    assert len(shown) == 5 and all(len(text) < 300 for text in shown)
    assert not any("pineapple secret" in text for text in shown)


def test_tool_limit_stops_before_extra_execution(indexed_loop, monkeypatch) -> None:
    executed = []
    original = tools.read

    def read(connection, path, scope=None):
        executed.append(path)
        return original(connection, path, scope)

    monkeypatch.setattr(tools, "read", read)
    scripted(monkeypatch, [
        ollama.Reply("", (ollama.ToolCall("read", {"path": "Alpha.md"}), ollama.ToolCall("read", {"path": "Beta.md"}))),
        ollama.Reply("SQLite stores passages [E1]."),
    ])
    turn = loop.answer(indexed_loop, "Read both notes", limits=loop.Limits(max_tools=1))
    assert turn.status == "limit"
    assert not executed and not turn.paths


def test_invalid_calls_consume_tool_limit(indexed_loop, monkeypatch) -> None:
    requests = scripted(monkeypatch, [
        ollama.Reply("", (ollama.ToolCall("missing", {}), ollama.ToolCall("read", {}))),
        ollama.Reply("", (ollama.ToolCall("survey", {}),)),
    ])
    turn = loop.answer(indexed_loop, "Read notes", limits=loop.Limits(max_tools=2))
    assert turn.status == "limit" and turn.model_calls == 2
    assert turn.tool_names == ("missing", "read")
    assert not turn.facts and not turn.evidence
    assert len([message for message in requests[1][0] if message["role"] == "tool"]) == 2


def test_last_model_call_omits_tools_and_limits_are_forwarded(indexed_loop, monkeypatch) -> None:
    read = ollama.Reply("", (ollama.ToolCall("read", {"path": "Alpha.md"}),))
    requests = scripted(monkeypatch, [read, read, read, ollama.Reply("SQLite stores passages [E1].")])
    turn = loop.answer(indexed_loop, "Read storage", limits=loop.Limits(timeout=17, num_predict=123))
    assert turn.status == "answered" and turn.model_calls == 4
    assert requests[-1][1] is None
    assert all(options["num_predict"] == 123 and options["timeout"] == 17 for _messages, _tools, options in requests)


def test_tools_on_final_call_are_not_executed(indexed_loop, monkeypatch) -> None:
    scripted(monkeypatch, [
        ollama.Reply("", (ollama.ToolCall("read", {"path": "Beta.md"}),)),
    ])
    turn = loop.answer(indexed_loop, "Read recovery", limits=loop.Limits(max_calls=1))
    assert turn.status == "limit" and not turn.paths


def test_generation_cutoff_is_reported_as_limit(indexed_loop, monkeypatch) -> None:
    scripted(monkeypatch, [ollama.Reply("Partial answer", done_reason="length")])
    assert loop.answer(indexed_loop, "Explain storage").status == "limit"


@pytest.mark.parametrize("scope,prior", [(None, ("Old question", "OLD ANSWER " * 10000)), (Scope(kinds=("research",)), ("Old question", "OLD ANSWER"))])
def test_oversized_or_scoped_prior_is_discarded(indexed_loop, monkeypatch, scope, prior) -> None:
    requests = scripted(monkeypatch, [ollama.Reply("Insufficient evidence: Need current notes.")])
    loop.answer(indexed_loop, "Current question", prior=prior, scope=scope)
    shown = json.dumps(requests[0][0])
    assert "OLD ANSWER" not in shown and "Old question" not in shown
    assert "Current question" in shown


def test_oversized_question_never_calls_model(indexed_loop, monkeypatch) -> None:
    requests = scripted(monkeypatch, [])
    turn = loop.answer(indexed_loop, "q" * 7000, context_size=2048)
    assert turn.status == "limit" and not requests


def test_source_budget_keeps_only_delivered_metadata(indexed_loop, monkeypatch) -> None:
    evidence = (
        Hit(100, "Alpha.md", "Storage decision", "Decision", "α" * 100000),
        Hit(101, "Beta.md", "Index recovery", "Recovery", "Never admitted"),
    )
    monkeypatch.setattr(tools, "read", lambda *_args, **_kwargs: tools.ToolResult("Indexed passages", evidence))
    scripted(monkeypatch, [
        ollama.Reply("", (ollama.ToolCall("read", {"path": "Alpha.md"}),)),
        ollama.Reply("This is an excerpt [E1]."),
    ])
    turn = loop.answer(indexed_loop, "Read storage")
    assert turn.evidence and turn.evidence[0].truncated
    assert turn.evidence[0].excerpt.endswith("[cut]")
    assert turn.paths == ("Alpha.md",)
    assert (turn.evidence[0].title, turn.evidence[0].heading) == ("Storage decision", "Decision")
    assert all(hit.path != "Beta.md" for hit in turn.evidence)


def test_huge_tool_arguments_cannot_overflow_next_request(indexed_loop, monkeypatch) -> None:
    requests = scripted(monkeypatch, [
        ollama.Reply("", (ollama.ToolCall("search", {"query": "é" * 100000}),)),
        ollama.Reply("Insufficient evidence: Tool argument was too large."),
    ])
    turn = loop.answer(indexed_loop, "Search storage")
    assert turn.status in {"limit", "insufficient_evidence"}
    assert all(len(json.dumps(messages).encode("utf-8")) < 8192 * 4 for messages, _tools, _options in requests)
    assert not turn.paths
