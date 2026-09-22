# Agent Brain

Chat with your Obsidian vault using local models. Your notes never leave the machine.

Two dependencies, one SQLite file, two commands to run.

## Setup

You need [Ollama](https://ollama.com) and [uv](https://docs.astral.sh/uv/).

```bash
ollama pull nomic-embed-text   # 274 MB, for search
ollama pull qwen3:4b           # 2.5 GB, for answers
```

Then, from this folder:

```bash
uv sync
uv run streamlit run app.py
```

The first launch reads your vault and builds the index. After that it starts instantly and
only re-reads notes you changed.

## Using it on another device

Clone this repo next to your vault and run the same two commands. Nothing else to configure.

```
~/Repos/
├── Obsidian_Vault/     your notes, synced by git
└── Agent_Brain/        this repo
```

The vault is found by looking for a folder containing `.obsidian`: first
`AGENT_BRAIN_VAULT` if you set it, then the folder you picked last time, then a single sibling
of this repo, then any parent of where you ran the command. If two vaults sit side by side it
asks rather than guessing, and remembers your answer.

The index is never committed. It lives in `~/.cache/agent_brain/`, one file per vault, because
it holds your note text and both repos have remotes. Each device builds its own in about a
minute.

## What it does

**Chat.** Ask a question, get an answer with the notes it came from. Every source shows its
path, heading, type and date, and whether words or meaning found it.

**Browse.** Filter across every note, or pick a folder and read anything in it. Each note
shows what links to it, what it links out to, and which notes are closest to it that it does
**not** already link to. Those are the links worth making.

**Graph.** Centre on a note and see what sits around it, one to three hops out along your own
wikilinks. Notes with more than 25 links are not expanded through, because a map-of-content
note otherwise drags in half the vault.

**Explore.** Groups the vault into themes and names each one, using the vectors already in
the index and a single model call per group. Below that, three things worth your attention:
concepts you linked but never wrote, notes you started and left nearly empty, and notes with
no links at all.

## Narrowing the search

348 of my 452 notes carry `type`, `tags` and `date_created` in their frontmatter. The sidebar
turns that into a type filter and a date range, and everything respects it: chat, themes, and
the counts.

Dates come in two shapes here. 147 notes write `2026-01-24 10:47:20` and 165 write
`Thursday 26th February 2026 13:38:57`, which is what Obsidian's template produces. Both are
read. Month names are matched from a table rather than through `strptime`, because `%B`
follows the machine's locale and the answer has to be the same on every device you clone to.

This exists because of a specific failure. Asked "what did I work on in March 2026?", the
search matched *March* and *2026* as words and returned January notes. A date range cannot be
wrong in that way, and it needs no date parsing to get right.

Notes with no date are excluded once a range is set, because an undated note cannot honestly
be said to fall inside it.

## How it finds things

Two searches run on every question and the results are merged by rank.

Word matching uses SQLite's FTS5. It is what finds an exact string: a command, an error code,
a person's name. Common words are dropped first, otherwise a note repeating "what" four times
outranks the note that answers you.

Meaning matching embeds the question and compares it against every passage. At this size that
is a few milliseconds of numpy, so there is no vector index to build or maintain. The same
vectors, averaged per note, drive the related-notes list and the theme clustering.

Merging uses rank rather than score. A BM25 score and a cosine similarity are not on the same
scale, and normalising one onto the other breaks whenever the distribution shifts. Rank needs
no calibration: a passage scores `1/(60 + rank)` in each list, so appearing in both beats
topping one.

Notes are split at their headings, and each piece carries its heading path. "use port 5432" is
ambiguous on its own, and the same text under `Postgres > Local setup` is not.

## Settings

Everything is in the sidebar. Models are listed from whatever Ollama already has, so there is
nothing to type and nothing to keep in sync.

Under **Settings**: how many passages go to the model, and the context window. The rebuild
button sits under the vault summary it invalidates.

## Development

```bash
uv run pytest
```

66 tests, no Ollama needed. The embedding call is stubbed, because what needs checking is the
code around the model rather than the model.

```
app.py            the Streamlit app, and the only entry point
brain/locate.py   which folder is the vault, and where the index goes
brain/notes.py    reading markdown and frontmatter, splitting notes into passages
brain/index.py    the SQLite schema, building and refreshing
brain/search.py   word matching, meaning matching, merging them, and related notes
brain/links.py    backlinks, neighbourhoods, and what the vault is missing
brain/themes.py   clustering the notes and naming the groups
brain/answer.py   the prompt, and fitting sources into the context window
brain/ollama.py   embedding and streaming, over the standard library
views/            one module per tab, plus notelist.py shared by all of them
```

Two dependencies: Streamlit, which brings numpy for the vector maths, and PyYAML for
frontmatter. Everything else is the standard library, including the Ollama calls and the
database.

There is no Streamlit inside `brain/` and no SQL inside `views/`. That line is worth keeping:
it means the logic can be read and tested without the UI, and the UI can be changed without
touching the logic.

## Read only

Agent Brain never writes to your vault. It reads notes and answers questions, so it cannot
corrupt anything mid-sync on a device where git is pulling.
