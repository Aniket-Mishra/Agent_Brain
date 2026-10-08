# Local evaluation results

Report dated 8 October 2026. The public fixture contains 12 invented notes and 20 questions. The fictional 2600-passage prototype mentioned inside the notes is not the size of this test.

Ollama 0.33.3, Qwen3 4B, nomic-embed-text, context 8192, and 800 generated tokens per chat call. Hardware: Apple M5 on macOS. Models ran on a separate loopback server with cloud features disabled.

Chat model digest: `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`. Embedding digest: `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`.

## Search with source locations

The latest run uses the same Search implementation as the app, including citation validation. Admitted passages carry their original line range and truncation flag.

```bash
OLLAMA_HOST=http://127.0.0.1:11435 uv run --locked python -m evals.run --mode baseline --model qwen3:4b --context 8192 --max-tokens 800
```

| Observation | Earlier Search | Search with source locations |
| --- | --- | --- |
| Completed answerable responses meeting literal fact checks | 12/12 | 12/12 |
| Median elapsed time across 20 questions | 11.58 seconds | 13.77 seconds |

These are single local runs. Literal matching does not establish support for every claim; `support_grade` remains unset pending human review. Known citations establish only that their evidence was admitted; this comparison does not demonstrate a lower hallucination rate.

The first citation trial produced 9 of 12 completed answerable responses meeting the literal checks, including a false abstention about FTS5. The final prompt restores original Search's passage-first order and removes the extra heading instruction; the rerun completed all 12. Generation is stochastic, so this result needs repetition with real questions.

| Case | Latest Search status |
| --- | --- |
| atlas-owner | answered |
| atlas-deadline | answered |
| orion-owner | answered |
| orion-export | answered |
| storage-choice | answered |
| word-search | answered |
| fusion-constant | answered |
| embedding-batch | answered |
| recover-model | answered |
| discovery-rehearsal | answered |
| compare-latency | answered |
| project-deadlines | answered |
| absent-password | insufficient_evidence |
| absent-price | limit |
| absent-benchmark | insufficient_evidence |
| scope-march | answered |
| scope-until | insufficient_evidence |
| followup-storage | limit |
| oversized-question | limit |
| hostile-instruction | answered |

The absent-price and follow-up cases still hit generation limits; the oversized question stops before retrieval or generation and the until-only scope admits no undated draft. The hostile-note case returned PINE-47 with a known citation. These cases cover the fixture only.

## Exploration remains experimental

The earlier full comparison met literal checks on 7 of 12 answerable questions for exploration, with a median of 29.89 seconds. A larger context of 16384 and generation budget of 1600 raised that result to 11 of 12, but both scoped cases hit generation limits.

During the citation follow-up, another full comparison again completed 7 of 12 answerable exploration responses meeting the checks, with a median of 20.29 seconds. That run preceded removal of the extra heading instruction; the final loop receives an additional native CLI check rather than a repeated full comparison. Search remains the default.

A Mistral Nemo 12B trial often omitted required citation identifiers despite retrieving useful notes. The validator rejected those responses. Both flows reject unknown citation identifiers and require citations for results marked `answered`; `insufficient_evidence` responses may omit them.

## Verification limits

Offline tests cover frontmatter offsets, repeated text, merged sections, fenced comments, short facts, Unicode, truncation, and admitted-versus-displayed evidence. Real CLI subprocesses and headless Streamlit interactions exercise the same source records.

The native CLI check returned `$40 [E1]` from `Budget Δ #1.md:8-23`; offline reading and note-title matching passed with fenced comments and short facts preserved. A one-token budget returned `limit`, and fixture hashes remained unchanged.

The native oversized-prompt probe returned HTTP 400 for a 10004-token request in a 2048-token context with truncation disabled. No silent clipping was accepted.

Reproduce the trials using the [fixture guide](README.md), then grade each claim against its cited passage. Timings vary with model loading and machine activity.
