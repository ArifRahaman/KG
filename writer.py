"""
All Neo4j writes.

Three tiers get written:
    (:Document)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(:__Entity__)
and typed relationships between entities, each carrying the chunk_ids that
asserted it.

Everything is MERGE, never CREATE, so re-ingesting the same file is safe.
"""

from __future__ import annotations

from contextlib import contextmanager

from neo4j import GraphDatabase

import config
from extract import Triple
from loaders import Document

# --- Schema ---------------------------------------------------------------
# Constraints create the backing index. Without them every MERGE scans all
# nodes instead of hitting an index -- the difference between minutes and
# hours on a real corpus.

CONSTRAINTS = [
    "CREATE CONSTRAINT doc_id IF NOT EXISTS "
    "FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
    "FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS "
    "FOR (e:__Entity__) REQUIRE e.id IS UNIQUE",
]

INDEXES = [
    "CREATE FULLTEXT INDEX entity_name IF NOT EXISTS "
    "FOR (e:__Entity__) ON EACH [e.name]",
]

# --- Queries --------------------------------------------------------------

WRITE_DOCUMENT = """
MERGE (d:Document {id: $doc_id})
SET   d.source_uri   = $source_uri,
      d.content_hash = $content_hash,
      d.ingested_at  = datetime()
WITH d
UNWIND $chunks AS ch
  MERGE (c:Chunk {id: ch.id})
  SET   c.text = ch.text, c.seq = ch.seq, c.content_hash = ch.content_hash
  MERGE (d)-[:HAS_CHUNK]->(c)
"""

# apoc.merge.* is required because Cypher cannot parameterize a label or a
# relationship type -- and ours arrive as data from the extractor.
WRITE_TRIPLES = """
UNWIND $rows AS row
  CALL apoc.merge.node(
      ['__Entity__', row.subject_type],
      {id: row.subject_id},
      {name: row.subject_name, created_at: datetime()},
      {}
  ) YIELD node AS s
  CALL apoc.merge.node(
      ['__Entity__', row.object_type],
      {id: row.object_id},
      {name: row.object_name, created_at: datetime()},
      {}
  ) YIELD node AS o
  CALL apoc.merge.relationship(
      s, row.predicate, {}, {created_at: datetime()}, o, {}
  ) YIELD rel
  // Provenance: append this chunk, keep the list deduplicated.
  SET rel.chunk_ids = apoc.coll.toSet(coalesce(rel.chunk_ids, []) + row.chunk_id)
  WITH s, o, row
  MATCH (c:Chunk {id: row.chunk_id})
  MERGE (c)-[:MENTIONS]->(s)
  MERGE (c)-[:MENTIONS]->(o)
"""

# OPTIONAL MATCH throughout so this still returns a row on an empty database,
# and so it works on every Neo4j 5.x/2025.x version (no CALL subqueries).
STATS = """
OPTIONAL MATCH (d:Document)
WITH count(d) AS documents
OPTIONAL MATCH (c:Chunk)
WITH documents, count(c) AS chunks
OPTIONAL MATCH (e:__Entity__)
WITH documents, chunks, count(e) AS entities
OPTIONAL MATCH (:__Entity__)-[r]->(:__Entity__)
WITH documents, chunks, entities, count(r) AS relationships
OPTIONAL MATCH (o:__Entity__) WHERE NOT (o)--()
RETURN documents, chunks, entities, relationships, count(o) AS orphans
"""

BY_TYPE = """
MATCH (:__Entity__)-[r]->(:__Entity__)
RETURN type(r) AS relationship, count(*) AS count
ORDER BY count DESC
"""


@contextmanager
def driver():
    drv = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
    )
    try:
        drv.verify_connectivity()
        yield drv
    finally:
        drv.close()


def _session(drv):
    return drv.session(database=config.NEO4J_DATABASE)


def check_apoc(drv) -> bool:
    with _session(drv) as session:
        result = session.run(
            "SHOW PROCEDURES YIELD name "
            "WHERE name = 'apoc.merge.node' RETURN count(*) AS n"
        )
        return result.single()["n"] > 0


def setup(drv) -> None:
    """Create constraints and indexes. Safe to run repeatedly."""
    with _session(drv) as session:
        for statement in CONSTRAINTS + INDEXES:
            session.run(statement)


def write_document(drv, document: Document) -> None:
    chunks = [
        {
            "id": c.id,
            "text": c.text,
            "seq": c.seq,
            "content_hash": c.content_hash,
        }
        for c in document.chunks
    ]
    with _session(drv) as session:
        session.run(
            WRITE_DOCUMENT,
            doc_id=document.id,
            source_uri=document.source_uri,
            content_hash=document.content_hash,
            chunks=chunks,
        )


def write_triples(drv, triples: list[Triple]) -> None:
    if not triples:
        return

    rows = [
        {
            "subject_id": t.subject_id,
            "subject_name": t.subject_name,
            "subject_type": t.subject_type,
            "predicate": t.predicate,
            "object_id": t.object_id,
            "object_name": t.object_name,
            "object_type": t.object_type,
            "chunk_id": t.chunk_id,
        }
        for t in triples
    ]

    size = config.WRITE_BATCH_SIZE
    with _session(drv) as session:
        for start in range(0, len(rows), size):
            session.run(WRITE_TRIPLES, rows=rows[start : start + size])


def graph_stats(drv) -> dict:
    with _session(drv) as session:
        summary = session.run(STATS).single().data()
        summary["by_type"] = [r.data() for r in session.run(BY_TYPE)]
        return summary


def wipe(drv) -> int:
    """
    Delete everything this pipeline created. Used by `reset`.

    Batched in Python rather than CALL {} IN TRANSACTIONS so it works on
    every Neo4j version and never blows the transaction heap.
    """
    total = 0
    with _session(drv) as session:
        while True:
            result = session.run(
                "MATCH (n) WHERE n:Document OR n:Chunk OR n:__Entity__ "
                "WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS deleted"
            )
            deleted = result.single()["deleted"]
            total += deleted
            if deleted == 0:
                return total
