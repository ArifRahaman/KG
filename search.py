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


ANSWER_PROMPT = """\
Answer the question using the retrieved passages and known facts below.

The passages are the source text. The facts come from a knowledge graph
built over that same text, and are there to fill in connections the passages
do not spell out. Prefer the passages where the two disagree.

Answer only from what is given. If it is not there, say so plainly.

### Question
{question}

### Retrieved passages
{passages}

### Known facts (knowledge graph)
{facts}
"""


# --- Retrieval -------------------------------------------------------------


def retrieve(drv, question: str, k_child: int = 0, k_parent: int = 0) -> list[dict]:
    """Embed the question, search children, return the deduplicated parents."""
    qvec = embed.embed_query(question)
    return drv.run(
        SEARCH,
        {
            "qvec": qvec,
            "k_child": k_child or config.SEARCH_CHILD_K,
            "k_parent": k_parent or config.SEARCH_PARENT_K,
        },
    )


def expand(drv, chunk_ids: list[str], limit: int = 40) -> list[dict]:
    """Pull structured facts for the entities the retrieved parents mention."""
    if not chunk_ids:
        return []
    return drv.run(EXPAND, {"chunk_ids": chunk_ids, "limit": limit})


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
        passages = retrieve(drv, question)

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
