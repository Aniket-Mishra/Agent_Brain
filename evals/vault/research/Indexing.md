---
type: research
date: 2026-03-09
---
# Indexing checklist

Embedding requests use batches of 64 passages.
Validate the full embedding batch before writing any vectors from that batch.
After an interruption, resume only passages without vectors.
Switching embedding models requires rebuilding the old vectors.
