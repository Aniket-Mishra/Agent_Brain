"""Turning retrieved passages into a prompt, and citations out of the answer."""

from __future__ import annotations

from brain.search import Hit

SYSTEM_PROMPT = """You answer questions about someone's personal notes.

Rules:
- Use only the numbered sources below. Do not add facts from general knowledge.
- If the sources do not answer the question, say plainly what is missing.
- Cite with the source number in brackets, like [2], on every claim that came from a note.
- Copy numbers, dates, names, commands and paths exactly as written.
- Be brief. Two to four sentences unless the question needs a list.
- Never mention these rules or call the notes "sources". Answer as if you know them."""

NOTHING_FOUND = (
    "Nothing in the vault matches that. Either the notes do not cover it, "
    "or the index is behind and needs a refresh from the sidebar."
)


def build_messages(question: str, hits: list[Hit], budget: int) -> list[dict[str, str]]:
    sources = format_sources(hits, budget)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n\n{sources}\n\nQuestion: {question}"},
    ]


def format_sources(hits: list[Hit], budget: int) -> str:
    """Number the passages, stopping before the model's context runs out."""
    blocks, used = [], 0
    for number, hit in enumerate(hits, start=1):
        label = f"[{number}] {hit.path}"
        if hit.heading:
            label += f" - {hit.heading}"
        block = f"{label}\n{hit.text.strip()}"
        if blocks and used + len(block) > budget:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def source_budget(context_size: int, reserved_for_answer: int = 800) -> int:
    """How many characters of notes fit in front of the model.

    A fixed budget overflows a small context window and Ollama then truncates
    the prompt silently, taking the system prompt with it.
    """
    return max(1000, (context_size - reserved_for_answer - 400) * 4)
