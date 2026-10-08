# Local evaluation results

Report dated 8 October 2026. The trial uses the public fixture: 12 invented notes and 20 questions. These results do not measure retrieval at 2600 passages; that number appears inside the fictional notes.

Ollama 0.33.3, Qwen3 4B, nomic-embed-text, context 8192, and 800 generated tokens per chat call. Hardware: Apple M5 on macOS; models ran on a separate loopback server with cloud features disabled.

Chat model digest: `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`. Embedding digest: `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`.

Command:

```bash
OLLAMA_HOST=http://127.0.0.1:11435 uv run --locked python -m evals.run --mode both --model qwen3:4b --context 8192 --max-tokens 800
```

| Observation | Fixed Search | Tool exploration |
| --- | --- | --- |
| Answerable questions meeting literal fact checks | 12/12 | 7/12 |
| Median elapsed time | 11.58 seconds | 29.89 seconds |

Literal matching is a debugging proxy. It does not establish that every claim is supported or correctly cited; `support_grade` remains unset in the raw records, pending human review.

The tool loop hits generation or context limits on several short questions, so Fixed Search remains the default; this trial does not justify promoting exploration. The UI and CLI expose the generation budget; larger contexts and token budgets cost more time.

| Case | Fixed Search status | Exploration status |
| --- | --- | --- |
| atlas-owner | answered | answered |
| atlas-deadline | answered | answered |
| orion-owner | answered | answered |
| orion-export | answered | answered |
| storage-choice | answered | answered |
| word-search | answered | answered |
| fusion-constant | answered | limit |
| embedding-batch | answered | limit |
| recover-model | answered | limit |
| discovery-rehearsal | answered | answered |
| compare-latency | answered | limit |
| project-deadlines | answered | limit |
| absent-password | answered | insufficient_evidence |
| absent-price | answered | limit |
| absent-benchmark | answered | insufficient_evidence |
| scope-march | answered | limit |
| scope-until | insufficient_evidence | limit |
| followup-storage | answered | limit |
| oversized-question | error | limit |
| hostile-instruction | answered | answered |

A separate Mistral Nemo 12B trial retrieved useful notes but often omitted required citation identifiers. Exploration rejected those answers. This is a model compatibility limit, so the app keeps fixed Search available.

Live CLI verification also covered a cited storage answer with context 16384 and 1600 generation tokens. Scoped search and survey worked, traversal was rejected, a one-token budget returned an explicit limit, and a missing chat model exited 1. SHA-256 hashes of all fixture notes were unchanged.

The native oversized-prompt check returned HTTP 400 for a 10004-token request in a 2048-token context with truncation disabled. No silent clipping was accepted.

Reproduce the trial using the [fixture guide](README.md), then grade the admitted and cited passages. Local timings vary with model loading and machine activity.
