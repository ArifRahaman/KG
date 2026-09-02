"""
Vector retrieval over the graph: search small, return big, expand.

    python search.py "why did TechCorp acquire CyberShield?"
    python search.py --passages "..."      show what was retrieved, no LLM

Three layers answer three different questions, and none of them replaces
another:

  ChildChunk vectors  find the right spot   (sharp, precise)
  Chunk text          give enough to read   (context, resolved referents)
  __Entity__ graph    supply structured facts and connected entities

The graph is lossy by construction -- a triple keeps that TechCorp acquired
CyberShield, but not why, not the caveats, not the hedges. That is what the
retrieved text is for.
"""

from __future__ import annotations

import re
import sys

import config
import embed
import writer

# --- Queries ---------------------------------------------------------------

# The child -> parent hop, and the whole reason this schema exists.
#
# Vector search lands on a ChildChunk, which is far too small to hand to an
# LLM -- it says "It opened an office there", not which company or where.
# One MATCH walks HAS_CHILD backwards to the parent, and the parent is what
# gets returned.
#
# The aggregation is doing two jobs at once. max(score) dedupes: if three
# adjacent children of one parent all match, the LLM still sees that parent
# exactly once instead of three near-identical copies. count(child) then
# rewards concentration -- a parent with three matching children is usually
# more relevant than one with a single marginally higher hit.
SEARCH = """
CALL db.index.vector.queryNodes('child_embedding', toInteger($k_child), $qvec)
YIELD node AS child, score
MATCH (parent:Chunk)-[:HAS_CHILD]->(child)
WITH parent, max(score) AS best, count(child) AS hits
WITH parent, best, hits, best + 0.02 * (hits - 1) AS rank
ORDER BY rank DESC
LIMIT toInteger($k_parent)
OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(parent)
RETURN parent.id   AS chunk_id,
       parent.text AS text,
       d.source_uri AS source,
       best AS score,
       hits AS matched_children
"""

# Structured facts for whatever the retrieved parents mention. The CASE pair
# is there because the match is undirected -- without it an inbound edge
# would be printed backwards.
EXPAND = """
UNWIND $chunk_ids AS cid
MATCH (c:Chunk {id: cid})-[:MENTIONS]->(e:__Entity__)
MATCH (e)-[r]-(n:__Entity__)
RETURN DISTINCT
    CASE WHEN startNode(r) = e THEN e.name ELSE n.name END AS source,
    type(r) AS relationship,
    CASE WHEN startNode(r) = e THEN n.name ELSE e.name END AS target
LIMIT toInteger($limit)
"""


# The lexical half of the hybrid. Same child -> parent hop, same dedupe, so
# the two runs produce directly comparable rows.
LEXICAL = """
CALL db.index.fulltext.queryNodes('child_text', $lucene)
YIELD node AS child, score
MATCH (parent:Chunk)-[:HAS_CHILD]->(child)
WITH parent, max(score) AS best, count(child) AS hits
ORDER BY best DESC
LIMIT toInteger($k_parent)
OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(parent)
RETURN parent.id   AS chunk_id,
       parent.text AS text,
       d.source_uri AS source,
       best AS score,
       hits AS matched_children
"""


DECOMPOSE_PROMPT = """\
Split the question below into the smallest set of standalone retrieval
queries that together cover everything it asks for.

A question with several parts usually has its answers in different parts of
a document. Embedding the whole question as one vector lets the loudest part
crowd out the rest, so each part needs its own search.

Rules:
- One query per line. No numbering, no preamble, no blank lines.
- Each query must stand alone: resolve pronouns, and repeat any identifier
  the original carried (e.g. "T-3") in every query that depends on it.
- At most 4. If the question only asks one thing, return it unchanged.

Question: {question}
"""


# --- Query planning --------------------------------------------------------


_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "what", "who", "which", "how",
    "does", "did", "has", "have", "been", "that", "this", "with", "from",
    "they", "why", "use", "used", "using", "these", "those", "their", "into",
}

# "T-3" and "T3" are the same label; a paper's prose and its tables rarely
# agree on which. Lucene's analyzer splits the hyphenated form into "t" and
# "3", so without the variant the identifier retrieves nothing at all.
_LABEL = re.compile(r"^([a-z]{1,3})-?(\d{1,3})$")


def plan(question: str) -> list[str]:
    """
    Turn one question into the queries needed to actually answer it.

    The original always stays in the list: decomposition can lose a framing
    that the whole question expressed and no part of it does.
    """
    response = embed.client.chat.completions.create(
        model=config.AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(question=question)}],
        temperature=0,
        max_tokens=256,
    )
    lines = [
        line.strip().lstrip("-*0123456789. ").strip()
        for line in response.choices[0].message.content.splitlines()
    ]

    queries, seen = [question], {question.lower()}
    for line in lines:
        if line and line.lower() not in seen:
            seen.add(line.lower())
            queries.append(line)
    return queries[: config.SEARCH_MAX_SUBQUERIES]


def _lucene(question: str) -> str:
    """Build a Lucene OR-query, quoting every term and expanding labels."""
    terms: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", question):
        token = raw.lower()
        if len(token) < 2 or token in _STOPWORDS:
            continue
        variants = {token}
        match = _LABEL.match(token)
        if match:
            stem, number = match.groups()
            variants |= {f"{stem}-{number}", f"{stem}{number}"}
        terms.extend(f'"{v}"' for v in variants)

    # Nothing survived filtering (a query of pure stopwords). Match nothing
    # rather than send Lucene an empty string, which is a syntax error.
    return " OR ".join(dict.fromkeys(terms)) or '"\uffff"'


def _rrf(runs: list[list[dict]], limit: int, k: int = 60) -> list[dict]:
    """
    Reciprocal rank fusion across every run.

    Vector scores and BM25 scores are on incomparable scales, so they cannot
    be added. Ranks can. A parent that places well in several runs beats one
    that tops a single run, which is exactly the behaviour a multi-part
    question needs.
    """
    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}

    for run in runs:
        for rank, row in enumerate(run, start=1):
            chunk_id = row["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            rows.setdefault(chunk_id, row)

    best = sorted(scores, key=lambda c: scores[c], reverse=True)[:limit]
    return [{**rows[c], "score": scores[c]} for c in best]


# --- Retrieval -------------------------------------------------------------


def retrieve(
    drv,
    question: str,
    k_child: int = 0,
    k_parent: int = 0,
    decompose: bool = True,
    queries: list[str] | None = None,
) -> list[dict]:
    """
    Plan the question, run vector and lexical search per sub-query, fuse.

    The budget scales with the number of sub-queries. Holding it at
    SEARCH_PARENT_K would defeat the decomposition: four sub-queries
    competing for five slots is the same crowding as before, just with more
    round trips.

    `queries` lets a caller reuse a plan it has already paid for.
    """
    k_child = k_child or config.SEARCH_CHILD_K
    k_parent = k_parent or config.SEARCH_PARENT_K

    if queries is None:
        queries = plan(question) if decompose else [question]
    runs: list[list[dict]] = []

    for query in queries:
        qvec = embed.embed_query(query)
        runs.append(
            drv.run(SEARCH, {"qvec": qvec, "k_child": k_child, "k_parent": k_parent})
        )
        try:
            runs.append(
                drv.run(LEXICAL, {"lucene": _lucene(query), "k_parent": k_parent})
            )
        except Exception:
            # The full-text index is created by `ingest.py setup`. A graph
            # built before it existed still works, just vector-only.
            pass

    limit = min(k_parent * len(queries), config.SEARCH_MAX_PARENTS)
    return _rrf(runs, limit)


def expand(drv, chunk_ids: list[str], limit: int = 40) -> list[dict]:
    """Pull structured facts for the entities the retrieved parents mention."""
    if not chunk_ids:
        return []
    return drv.run(EXPAND, {"chunk_ids": chunk_ids, "limit": limit})


ANSWER_PROMPT = """\
Answer the question using the retrieved passages and known facts below.

The passages are the source text. The facts come from a knowledge graph
built over that same text, and are there to fill in connections the passages
do not spell out. Prefer the passages where the two disagree.

Use only what is given here. Do not draw on outside knowledge.

A multi-part question is answered part by part. Answer every part the
passages support, even if others are unsupported -- a partial answer with
its gaps named is useful, and refusing the whole question because one part
is missing is not. Only say nothing was found if no part is supported.

Close with a line beginning "Not covered:" naming any part you could not
answer, or omit the line entirely if you answered all of them.

### Question
{question}

### Retrieved passages
{passages}

### Known facts (knowledge graph)
{facts}
"""


def _format_passages(passages: list[dict]) -> str:
    if not passages:
        return "(none)"
    blocks = []
    for p in passages:
        source = p.get("source") or "unknown"
        blocks.append(f"[{source}, chunk {p['chunk_id']}]\n{p['text']}")
    return "\n\n".join(blocks)


def _format_facts(facts: list[dict]) -> str:
    if not facts:
        return "(none)"
    return "\n".join(
        f"{f['source']}  {f['relationship']}  {f['target']}" for f in facts
    )


def answer(question: str, passages: list[dict], facts: list[dict]) -> str:
    response = embed.client.chat.completions.create(
        model=config.AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": ANSWER_PROMPT.format(
                    question=question,
                    passages=_format_passages(passages),
                    facts=_format_facts(facts),
                ),
            }
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def ask(drv, question: str) -> tuple[str, list[dict], list[dict]]:
    passages = retrieve(drv, question)
    facts = expand(drv, [p["chunk_id"] for p in passages])
    return answer(question, passages, facts), passages, facts


# --- CLI -------------------------------------------------------------------


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    passages_only = "--passages" in sys.argv[1:]

    if not args:
        print(__doc__)
        return

    question = " ".join(args)

    with writer.driver() as drv:
        queries = plan(question)
        if len(queries) > 1:
            print(f"\nPlanned {len(queries)} sub-queries:")
            for q in queries:
                print(f"  - {q}")

        passages = retrieve(drv, question, queries=queries)

        if not passages:
            print("\nNothing retrieved. Is anything ingested with embeddings?")
            print("Check with:  python ingest.py stats")
            return

        print(f"\nRetrieved {len(passages)} parent chunk(s)")
        print("-" * 60)
        for p in passages:
            preview = " ".join(p["text"].split())[:110]
            print(
                f"  {p['chunk_id']}  score={p['score']:.3f}  "
                f"children_matched={p['matched_children']}"
            )
            print(f"    {preview}...")

        facts = expand(drv, [p["chunk_id"] for p in passages])
        print(f"\n{len(facts)} fact(s) from the graph")

        if passages_only:
            print("\n--- passages ---")
            print(_format_passages(passages))
            print("\n--- facts ---")
            print(_format_facts(facts))
            return

        print("-" * 60)
        print(f"\nAnswer: {answer(question, passages, facts)}\n")


if __name__ == "__main__":
    main()
