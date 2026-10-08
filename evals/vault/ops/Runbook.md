---
type: operations
date: 2026-02-12
---
# Recovery runbook

The recovery rehearsal is called Cedar Drill.
Before a rehearsal, copy the fixture export to backups/atlas.jsonl.
The cache is disposable: remove the test cache, then rebuild it from unchanged note files.
For a missing model, install nomic-embed-text before rebuilding the cache.
