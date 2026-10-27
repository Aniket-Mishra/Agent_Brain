import pytest

from brain import ollama, themes


@pytest.mark.parametrize("content,expected", [
    ('"Storage"', "Storage"), ('"Storage"\nIgnored', 'Storage"'), ("x" * 100, "x" * 60),
])
def test_theme_naming_preserves_samples_limits_and_label(monkeypatch, content, expected) -> None:
    members = [f"Daily/{number:02}.md" for number in range(25)]
    titles = {path: f"Declared {number}" for number, path in enumerate(members)}
    del titles[members[0]]
    titles[members[1]] = ""
    sent = []

    def complete(messages, model, **options):
        sent.append((messages, model, options))
        return ollama.Reply(content)

    monkeypatch.setattr(ollama, "complete", complete)
    assert themes.name_group(members, titles, "local") == expected
    sample = ["00", "", *(f"Declared {number}" for number in range(2, 20))]
    assert sent == [(
        [{"role": "user", "content": themes.NAMING_PROMPT.format(titles="\n".join(sample))}],
        "local", {"context_size": 4096, "num_predict": 800, "timeout": 600},
    )]


@pytest.mark.parametrize("reply", [
    ollama.Reply(""), ollama.Reply(" \n"), ollama.Reply('""'),
    ollama.Reply("Incomplete name", done_reason="length"), ollama.OllamaError("model failed"),
])
def test_theme_naming_preserves_folder_fallback(monkeypatch, reply) -> None:
    def complete(*_args, **_options):
        if isinstance(reply, ollama.OllamaError):
            raise reply
        return reply

    monkeypatch.setattr(ollama, "complete", complete)
    assert themes.name_group(["Daily/a.md", "Daily/b.md"], {}, "local") == "Daily"
