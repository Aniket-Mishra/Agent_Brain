"""Compare answer flows against public fixtures; support grading stays manual."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import time

from brain import index, ollama
from brain.answer import NOTHING_FOUND, build_messages, format_sources, source_budget
from brain.search import Scope, search

FIXTURE = Path(__file__).parent
CASES = json.loads((FIXTURE / "cases.json").read_text(encoding="utf-8"))


def baseline(connection, case: dict, question: str, options, scope: Scope) -> dict:
    hits = search(connection, question, options.embed, limit=options.passages, scope=scope)
    if not hits:
        return {
            "text": NOTHING_FOUND, "status": "insufficient_evidence", "evidence": (),
            "paths": (), "cited": (), "facts": (), "cited_facts": (),
            "tool_names": ("search",), "model_calls": 0,
            "prompt_eval_count": 0, "eval_count": 0,
        }
    budget = source_budget(options.context, options.max_tokens)
    formatted = format_sources(hits, budget)
    admitted_count = next(
        count for count in range(1, len(hits) + 1)
        if format_sources(hits[:count], budget) == formatted
    )
    admitted = tuple(hits[:admitted_count])
    reply = ollama.complete(
        build_messages(question, hits, budget), options.model, options.context,
        num_predict=options.max_tokens,
    )
    return {
        "text": reply.content, "status": "limit" if reply.done_reason == "length" else "answered", "evidence": admitted,
        "paths": tuple(dict.fromkeys(hit.path for hit in admitted)),
        "cited": tuple(int(number) for number in re.findall(r"\[(\d+)\]", reply.content)),
        "facts": (), "cited_facts": (),
        "tool_names": ("search",), "model_calls": 1,
        "prompt_eval_count": reply.prompt_eval_count, "eval_count": reply.eval_count,
    }


def agent(connection, case: dict, question: str, options, scope: Scope) -> dict:
    from brain.loop import Limits, answer

    prior = tuple(message["content"] for message in case.get("prior", [])) or None
    turn = answer(
        connection, question, prior=prior, chat_model=options.model,
        embed_model=options.embed, passages=options.passages,
        context_size=options.context, scope=scope,
        limits=Limits(num_predict=options.max_tokens),
    )
    names = (
        "text", "status", "evidence", "paths", "cited", "facts", "cited_facts",
        "tool_names", "model_calls",
        "prompt_eval_count", "eval_count",
    )
    return {name: getattr(turn, name) for name in names}


def record_result(case: dict, result: dict) -> dict:
    evidence = []
    for number, hit in enumerate(result["evidence"], start=1):
        passage = asdict(hit)
        passage["found_by"] = sorted(hit.found_by)
        passage["evidence_id"] = number
        evidence.append(passage)
    result["evidence"] = evidence
    result["facts"] = [asdict(fact) for fact in result["facts"]]
    result["tool_executions"] = len(result["tool_names"])
    result["expected_path_proxy"] = set(case["expected_paths"]).issubset(result["paths"])
    result["literal_fact_proxy"] = {
        fact: fact.casefold() in result["text"].casefold() for fact in case["required_facts"]
    }
    result["forbidden_literal_proxy"] = {
        fact: fact.casefold() in result["text"].casefold() for fact in case.get("forbidden_facts", [])
    }
    result["citations_known"] = (
        all(1 <= number <= len(evidence) for number in result["cited"])
        and all(1 <= number <= len(result["facts"]) for number in result["cited_facts"])
    )
    result["support_grade"] = None
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--embed", default="nomic-embed-text")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--passages", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--case", choices=[case["id"] for case in CASES])
    parser.add_argument("--mode", choices=("baseline", "agent", "both"), default="both")
    options = parser.parse_args()
    if options.context < 2048 or options.passages < 1 or options.max_tokens < 1:
        parser.error("--context must be at least 2048; --passages and --max-tokens must be positive")
    metadata = {
        "ollama_version": ollama.request("/api/version", None)["version"],
        "chat_model": options.model, "chat_digest": ollama.model_digest(options.model),
        "embedding_model": options.embed, "embedding_digest": ollama.model_digest(options.embed),
        "context_size": options.context, "passages": options.passages,
        "max_tokens": options.max_tokens,
        "python_version": platform.python_version(), "platform": platform.platform(),
    }
    cases = [case for case in CASES if not options.case or case["id"] == options.case]
    modes = ("baseline", "agent") if options.mode == "both" else (options.mode,)
    with tempfile.TemporaryDirectory(prefix="agent-brain-eval-") as directory:
        vault = shutil.copytree(FIXTURE / "vault", Path(directory) / "vault")
        connection = index.connect(Path(directory) / "index.db")
        try:
            index.refresh(connection, vault, options.embed, model_digest=metadata["embedding_digest"])
            for case in cases:
                scope = Scope(**{**case["scope"], "kinds": tuple(case["scope"].get("kinds", ()))})
                question = case["question"] * case.get("repeat", 1)
                for mode in modes:
                    started = time.perf_counter()
                    try:
                        result = (baseline if mode == "baseline" else agent)(
                            connection, case, question, options, scope
                        )
                        result = record_result(case, result)
                    except (ollama.OllamaError, ValueError) as error:
                        result = {"status": "error", "error": str(error), "support_grade": None}
                    record = {
                        **metadata, **case, **result, "mode": mode,
                        "question_bytes": len(question.encode("utf-8")),
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    }
                    print(json.dumps(record, ensure_ascii=False), flush=True)
        finally:
            connection.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ollama.OllamaError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
