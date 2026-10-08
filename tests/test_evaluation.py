"""Offline fixture checks; model prose is graded separately."""

from collections import Counter
from datetime import date

from brain.notes import chunk_note, find_notes, read_note
from evals.run import CASES, FIXTURE


def test_cases_reference_readable_fixture_notes() -> None:
    vault = FIXTURE / "vault"
    parsed = [read_note(path, vault) for path in find_notes(vault)]
    notes = {note.path: note for note in parsed}
    assert (vault / ".obsidian").is_dir()
    assert 8 <= len(notes) <= 12
    assert all(chunk_note(note) for note in notes.values())
    assert len(CASES) == len({case["id"] for case in CASES}) == 20
    assert Counter(case["category"] for case in CASES) == {
        "answerable": 12, "absent": 3, "scope": 2, "followup": 1,
        "oversized": 1, "hostile": 1,
    }
    for case in CASES:
        assert case["question"].strip()
        assert case["required_facts"] or case.get("expected_abstention")
        assert set(case["expected_paths"]).issubset(notes)
        if "prior" in case:
            assert [message["role"] for message in case["prior"]] == ["user", "assistant"]
            assert all(isinstance(message["content"], str) for message in case["prior"])
        assert set(case["scope"]) <= {"kinds", "since", "until"}
        for field in ("since", "until"):
            if field in case["scope"]:
                date.fromisoformat(case["scope"][field])
        expected_text = "\n".join(
            chunk.text for path in case["expected_paths"] for chunk in chunk_note(notes[path])
        )
        assert all(fact.casefold() in expected_text.casefold() for fact in case["required_facts"])
