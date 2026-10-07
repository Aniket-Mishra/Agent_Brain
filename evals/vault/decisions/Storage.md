---
type: research
date: 2026-02-10
---
# Storage decision

Atlas uses SQLite. SQLite was chosen because one disposable local file needs no server.
Word matching uses FTS5. Vector search uses exact cosine similarity across all passages.
The supported prototype has 2600 passages. Recovery instructions live in [[Runbook]].
