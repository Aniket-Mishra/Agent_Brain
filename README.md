# Agent Brain

Chat with your Obsidian vault using local models. Your notes never leave the machine.

One dependency, one SQLite file, two commands to run.

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

**Chat.** Ask a question, get an answer with the notes it came from. Every source is listed
with its path and heading, and you can read the passage without leaving the answer.

**Browse.** Filter across every note, or pick a folder and read anything in it. Each note
shows what links to it and what it links out to.

**Graph.** Centre on a note and see what sits around it, one to three hops out along your own
wikilinks. Notes with more than 25 links are not expanded through, because a map-of-content
note otherwise drags in half the vault.

## How it finds things

Two searches run on every question and the results are merged by rank.

Word matching uses SQLite's FTS5. It is what finds an exact string: a command, an error code,
a person's name. Common words are dropped first, otherwise a note repeating "what" four times
outranks the note that answers you.

Meaning matching embeds the question and compares it against every passage. At this size that
is a few milliseconds of numpy, so there is no vector index to build or maintain.

Merging uses rank rather than score. A BM25 score and a cosine similarity are not on the same
scale, and normalising one onto the other breaks whenever the distribution shifts. Rank needs
no calibration: a passage scores `1/(60 + rank)` in each list, so appearing in both beats
topping one.

Notes are split at their headings, and each piece carries its heading path. "use port 5432" is
ambiguous on its own, and the same text under `Postgres > Local setup` is not.

## Settings

Everything is in the sidebar. Models are listed from whatever Ollama already has, so there is
nothing to type and nothing to keep in sync.

Under **Settings**: how many passages go to the model, the context window, and a button to
rebuild the index from scratch.

## Development

```bash
uv run pytest
```

36 tests, no Ollama needed. The embedding call is stubbed, because what needs checking is the
code around the model rather than the model.

```
app.py            the Streamlit app, and the only entry point
brain/locate.py   which folder is the vault, and where the index goes
brain/notes.py    reading markdown, and splitting it into passages
brain/index.py    the SQLite schema, building and refreshing
brain/search.py   word matching, meaning matching, and merging them
brain/links.py    backlinks and neighbourhoods
brain/answer.py   the prompt, and fitting sources into the context window
brain/ollama.py   embedding and streaming, over the standard library
views/            one module per tab
```

Streamlit is the only dependency. It brings numpy, which is all the vector maths needs.
Everything else is the standard library, including the Ollama calls.

## Read only

Agent Brain never writes to your vault. It reads notes and answers questions, so it cannot
corrupt anything mid-sync on a device where git is pulling.
