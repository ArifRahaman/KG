# mini KG — the whole idea in one file

This is a **lite version of the knowledge graph in the folder above**, meant
to be *read*, not deployed. The full project is split across a dozen files and
needs a running Neo4j database. This one is a single script,
[`mini_kg.py`](mini_kg.py), that does the same core thing you can read
top to bottom in a few minutes.

```
   your file  ─►  chunks  ─►  LLM extraction  ─►  graph  ─►  query
                  (free)      (facts as triples)  (JSON)
```

The pipeline is identical to the real one. Only the **storage** changes: the
full system writes to Neo4j; this one keeps the graph in a Python dict and
saves it to `graph.json`. That single swap is what lets you run it with no
database, no Docker, no APOC — just an Azure OpenAI key (the same `.env` the
full project already uses, one folder up).

---

## The one big idea: a fact is a triple

A knowledge graph stores facts as **(subject) —predicate→ (object)**:

```
(TechCorp) ─FOUNDED_BY→ (Sarah Chen)
(TechCorp) ─ACQUIRED→   (DataFlow Systems)
(Sarah Chen) ─CEO_OF→   (TechCorp)
```

Each subject and object is a **node** (an entity); each predicate is an
**edge** (a relationship). Build enough of these from a document and you can
walk the connections instead of re-reading the text. That is the whole trick —
everything below is machinery to get facts *into* that shape and *back out*.

---

## Run it

From inside this folder:

```bash
# 1. Read a file, chunk it, extract facts, save the graph  (uses the LLM)
python mini_kg.py build ../recovered/doc0_tmpxvcnayr8.txt

# 2. See what came out                                     (no keys needed)
python mini_kg.py stats

# 3. Inspect one entity and everything touching it         (no keys needed)
python mini_kg.py show TechCorp

# 4. Ask a question, answered only from the graph's facts  (uses the LLM)
python mini_kg.py ask "Who founded the company DataFlow?"

# 5. Draw the graph in a browser                           (no keys needed)
python mini_kg.py export      # then open graph.html
```

`build` and `ask` call Azure OpenAI. `stats`, `show` and `export` only read
`graph.json`, so they work offline once a graph exists.

Example of `show`:

```
TechCorp Inc.  (Organization)
-----------------------------
  -> FOUNDED_BY        Sarah Chen
  -> ACQUIRED          DataFlow Systems
  <- CEO_OF            Sarah Chen

  source: "TechCorp Inc. was founded by Sarah Chen and Michael Rodriguez..."
```

`->` means TechCorp is the subject of the fact; `<-` means it is the object
of someone else's. The `source:` line is **provenance** — every edge
remembers which chunk asserted it, so any fact traces back to the sentence it
came from.

---

## How the code maps to the full project

`mini_kg.py` is one file, but it is cut into sections that line up with the
real modules. Read a section here, then open its bigger sibling upstairs.

| Section in `mini_kg.py` | Full project | What changed in the lite version |
|---|---|---|
| **0. Config + client** | `config.py`, `extract.py` | Same `.env`, same Azure client. Built lazily so read-only commands need no keys. |
| **1. Load + chunk** | `loaders.py` | Kept paragraph-packing. Dropped PDF/CSV/JSON readers and the vector-search "child" layer. Any text file works. |
| **2. Extract triples** | `extract.py` + `ontology.py` | Same dynamic-ontology prompt, same deterministic entity id (so `Acme Corp` = `Acme Corporation`). Dropped pydantic + token accounting. |
| **3. The graph** | `writer.py` + **Neo4j** | **The big swap.** Three Python dicts + a JSON file replace the database. `add_triple` is the equivalent of Neo4j `MERGE`. |
| **4. `ask`** | `search.py` | Same graph→facts→LLM answer, but **without vector search**: small graphs just hand the LLM every fact. |
| **4. `export`** | *(none)* | Extra: a browser view so you can *see* the graph. |
| CLI | `ingest.py` | Same `build / stats / show / …` command shape. |

---

## What the lite version deliberately leaves out

These are exactly the things that make the full project heavier — and are
worth understanding *after* the core clicks:

- **Neo4j** — a real graph database with indexes and a query language (Cypher).
  Needed once a graph is too big for one JSON file.
- **Embeddings + vector search** (`embed.py`, the "child" chunks) — how the
  full system finds the *relevant* passages before answering, instead of
  dumping every fact into the prompt.
- **Provenance at scale, incremental re-ingest, an HTTP API** (`api.py`,
  `chat.py`) — production concerns, not concepts.

If the mini version makes sense, the full one upstairs is the same shape with
these three worries added back in.
