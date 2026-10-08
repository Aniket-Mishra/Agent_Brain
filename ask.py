"""Shell entry point for read-only vault tools and bounded local answers."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from contextlib import closing
from datetime import date
from pathlib import Path

from brain import index, loop, ollama, tools
from brain.locate import database_path, find_vault, is_vault
from brain.search import Scope


def calendar_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from error
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD")
    return value


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    arguments = argparse.ArgumentParser(description=__doc__)
    limits = loop.Limits()
    commands = arguments.add_subparsers(dest="command", required=True)
    for name in ("search", "read", "backlinks", "related", "survey", "ask"):
        command = commands.add_parser(name)
        command.add_argument("--embed", help="embedding model; defaults to index metadata, then environment")
        command.add_argument("--kind", action="append", default=[])
        command.add_argument("--since", type=calendar_date)
        command.add_argument("--until", type=calendar_date)
        if name in {"read", "backlinks", "related"}:
            command.add_argument("path")
        if name in {"search", "ask"}:
            command.add_argument("question", nargs="+")
            command.add_argument("--passages", type=int, choices=range(3, 16), default=8)
        if name == "ask":
            command.add_argument("--model", default=os.environ.get("AGENT_BRAIN_CHAT_MODEL", "qwen3:4b"))
            command.add_argument("--context", type=int, choices=(4096, 8192, 16384, 32768), default=8192)
            command.add_argument("--prior-user")
            command.add_argument("--prior-assistant")
            command.add_argument("--max-calls", type=positive_integer, default=limits.max_calls)
            command.add_argument("--max-tools", type=positive_integer, default=limits.max_tools)
            command.add_argument("--max-tokens", type=positive_integer, default=limits.num_predict)
            command.add_argument("--timeout", type=positive_integer, default=limits.timeout)
    return arguments


def resolve_vault() -> Path:
    named = os.environ.get("AGENT_BRAIN_VAULT")
    vault = Path(named).expanduser().resolve() if named else find_vault()
    if vault is None or not is_vault(vault):
        raise ValueError("Set AGENT_BRAIN_VAULT to a folder that contains .obsidian")
    return vault


def run(arguments: argparse.Namespace) -> None:
    if arguments.since and arguments.until and arguments.since > arguments.until:
        raise ValueError("--since must not be after --until")
    prior = None
    if arguments.command == "ask":
        if (arguments.prior_user is None) != (arguments.prior_assistant is None):
            raise ValueError("Supply --prior-user and --prior-assistant together")
        if arguments.prior_user is not None:
            prior = (arguments.prior_user, arguments.prior_assistant)
    question = " ".join(arguments.question).strip() if arguments.command in {"search", "ask"} else ""
    if arguments.command in {"search", "ask"} and not question:
        raise ValueError("Question must not be empty")
    if arguments.command == "search":
        tools.validate_query(question)
    vault = resolve_vault()
    scope = Scope(tuple(arguments.kind), arguments.since or "", arguments.until or "")
    with closing(index.connect(database_path(vault))) as connection:
        stored = index.read_meta(connection, "embedding_model")
        embed_model = arguments.embed or stored or os.environ.get("AGENT_BRAIN_EMBED_MODEL", "nomic-embed-text")
        current = index.is_current(connection, vault, embed_model)
        digest = None
        if not current or arguments.command in {"search", "related", "ask"}:
            digest = ollama.model_digest(embed_model)
            current = index.is_current(connection, vault, embed_model, model_digest=digest)
        if not current:
            if stored is not None and not index.same_model(stored, embed_model):
                print(f"rebuilding the index for {embed_model}", file=sys.stderr)
            index.refresh(
                connection, vault, embed_model,
                lambda stage, done, total: print(f"{stage} {done}/{total}", file=sys.stderr),
                model_digest=digest,
            )
            if not index.is_current(connection, vault, embed_model, model_digest=digest):
                raise RuntimeError("Vault changed during indexing. Run the command again.")
        if arguments.command == "ask":
            turn = loop.answer(
                connection, question, prior, arguments.model, embed_model,
                arguments.passages, arguments.context, scope,
                lambda name: print(f"tool {name}", file=sys.stderr),
                limits=loop.Limits(arguments.max_calls, arguments.max_tools, arguments.max_tokens, arguments.timeout),
            )
            print(turn.text)
            if turn.status != "answered":
                print(f"Status: {turn.status}")
            print("\nRetrieved evidence:")
            for number, hit in enumerate(turn.evidence, 1):
                cited = " (cited)" if number in turn.cited else ""
                print(f"[E{number}] {hit.path} | {hit.heading}{cited}")
            for number, fact in enumerate(turn.facts, 1):
                cited = " (cited)" if number in turn.cited_facts else ""
                print(f"[I{number}]{cited}\n{fact.text}")
            return
        if arguments.command == "search":
            result = tools.search(connection, question, embed_model, arguments.passages, scope)
        elif arguments.command == "survey":
            result = tools.survey(connection, scope)
        else:
            result = getattr(tools, arguments.command)(connection, arguments.path, scope)
        print(tools.format_result(result))


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = parser().parse_args(argv)
        run(arguments)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    except (ollama.OllamaError, sqlite3.Error, OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
