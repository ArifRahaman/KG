# Step 1 — Documents to Knowledge Graph

Upload a PDF, CSV, JSON, or text file. Get a queryable graph in Neo4j.

```
your file  ->  chunks  ->  LLM extraction  ->  Neo4j
                (free)      (ontology-constrained)
```

---

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure**

Copy `.env.example` to `.env` and fill in your Neo4j details and Anthropic API key.

```bash
cp .env.example .env
```

**3. Make sure APOC is installed on your Neo4j**

APOC is required, not optional. Cypher cannot parameterize node labels or
relationship types, and ours arrive as data from the extractor.

| Where | How |
|---|---|
| Neo4j Desktop | Open your database → **Plugins** → **APOC** → Install → restart |
| Docker | add `-e NEO4J_PLUGINS='["apoc"]'` |
| Aura | APOC Core is preinstalled |

**4. Create constraints and indexes** (once)

```bash
python ingest.py setup
```

This verifies APOC is present and creates the uniqueness constraints. Without
them every `MERGE` scans all nodes instead of hitting an index — the
difference between minutes and hours on a real corpus.

---

## Use

```bash
# See how a file will be chunked -- no LLM calls, nothing written
python ingest.py mydoc.pdf --dry-run

# Build the graph
python ingest.py mydoc.pdf

# What's in the graph now
python ingest.py stats

# Start over
python ingest.py reset
```

**Always dry-run a new file type first.** It costs nothing and shows you
exactly what the model will see.

### Supported files

| Type | How it's chunked |
|---|---|
| `.pdf` | Text extracted per page, then packed into ~2000-char chunks on paragraph boundaries |
| `.txt` `.md` | Paragraph split, then packed |
| `.csv` `.tsv` | Each row becomes a record; records are packed several per chunk |
| `.json` `.jsonl` | Each object becomes a record; packed the same way |

Scanned PDFs will not work — `pypdf` reads embedded text, not images of text.
Those need OCR first.

---

## See your graph

Open Neo4j Browser and run:

```cypher
MATCH (n:__Entity__)-[r]->(m:__Entity__) RETURN n, r, m LIMIT 100
```

Trace a fact back to the sentence it came from:

```cypher
MATCH (s:__Entity__)-[r]->(o:__Entity__)
UNWIND r.chunk_ids AS cid
MATCH (c:Chunk {id: cid})
RETURN s.name, type(r), o.name, c.text
LIMIT 25
```

Every relationship carries the `chunk_ids` that asserted it. That is what
gives you citations and lets you audit a wrong answer back to its source.

---

## Change the ontology

**`ontology.py` is the file you edit.** It ships with a general-purpose
starter set (Person, Organization, Location, Product, Event, Concept).

A tight domain ontology extracts far better than a generic one. Once you know
your domain, replace the types with your real ones and update `ALLOWED` — the
set of legal `(subject, predicate, object)` combinations.

`ALLOWED` is the real schema. It is not enough to list labels and relationship
types separately; what matters is which combinations are legal. Anything the
model returns that is not in that set is dropped and counted, never written.

---

## What the graph looks like

```
(:Document)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(:Person:__Entity__)
                                                      |
                                                 [:WORKS_AT]
                                                      v
                                              (:Organization:__Entity__)
```

Every entity carries `:__Entity__` **plus** its domain label. That way one
index covers the whole knowledge tier no matter how many types you add later.

---

## Notes

**Re-running is safe.** Everything is `MERGE`, never `CREATE`, and entity IDs
are a deterministic hash of the normalized name. Ingest the same file twice
and you get the same graph, not a duplicate one.

**Name variants collapse automatically.** `Acme Corp`, `Acme Corporation`,
`ACME CORP` and `Acme Corp Inc` all resolve to one node. This is basic
resolution only — fuzzy matching (`Acme` vs `Acme Systems`) is a later step.

**Cost.** The ontology sits in a cached system prompt, so after the first call
it bills at roughly 10%. The run summary prints token counts including cache
hits so you can see it working. Tune `KG_EFFORT` in `.env` if extraction
quality or cost needs adjusting.

---

## Not in this step

Deliberately left out to keep step 1 shippable:

- **Embeddings and vector search** — needed for retrieval, not for building the graph
- **Fuzzy entity resolution** — GDS KNN over name embeddings
- **Community detection** — for corpus-wide thematic questions
- **Incremental re-ingest** — currently a changed document re-extracts in full
