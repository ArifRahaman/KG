"""
Build a knowledge graph from a file.

    python ingest.py setup              create constraints + indexes (run once)
    python ingest.py <file>             ingest a PDF / CSV / JSON / TXT / MD
    python ingest.py <file> --dry-run   chunk only, no LLM, no writes
    python ingest.py stats              what is in the graph now
    python ingest.py reset              delete everything (asks first)
"""

from __future__ import annotations

import sys
import time

import loaders
import writer
from extract import ExtractionStats, extract_chunk


def _banner(text: str) -> None:
    print(f"\n{text}")
    print("-" * len(text))


def cmd_setup() -> None:
    with writer.driver() as drv:
        _banner("Setup")
        if not writer.check_apoc(drv):
            sys.exit(
                "APOC is not installed on this Neo4j instance.\n\n"
                "APOC is required -- Cypher cannot parameterize node labels or\n"
                "relationship types, and ours come from the extractor as data.\n\n"
                "  Neo4j Desktop : open your database -> Plugins -> APOC -> Install\n"
                "  Docker        : -e NEO4J_PLUGINS='[\"apoc\"]'\n"
                "  Aura          : APOC Core is preinstalled (check your URI)\n"
            )
        print("  APOC found")
        writer.setup(drv)
        print("  Constraints and indexes created")
        print("\nReady. Now run:  python ingest.py <your-file.pdf>")


def cmd_stats() -> None:
    with writer.driver() as drv:
        stats = writer.graph_stats(drv)
        _banner("Graph")
        print(f"  Documents      {stats['documents']:>8,}")
        print(f"  Chunks         {stats['chunks']:>8,}")
        print(f"  Entities       {stats['entities']:>8,}")
        print(f"  Relationships  {stats['relationships']:>8,}")

        if stats["orphans"]:
            print(
                f"\n  {stats['orphans']:,} orphan entities "
                "(extracted but connected to nothing)"
            )

        if stats["by_type"]:
            _banner("Relationships by type")
            for row in stats["by_type"]:
                print(f"  {row['relationship']:<24} {row['count']:>6,}")


def cmd_reset() -> None:
    answer = input("Delete all Documents, Chunks and Entities? [y/N] ").strip()
    if answer.lower() != "y":
        print("Cancelled.")
        return
    with writer.driver() as drv:
        writer.wipe(drv)
        print("Graph cleared.")


def cmd_ingest(path: str, dry_run: bool) -> None:
    started = time.time()

    _banner(f"Loading {path}")
    document = loaders.load(path)
    print(f"  {len(document.chunks)} chunks")

    if dry_run:
        _banner("Dry run -- first 3 chunks")
        for chunk in document.chunks[:3]:
            preview = chunk.text[:300].replace("\n", " ")
            print(f"\n  [{chunk.seq}] {preview}...")
        print("\nNo LLM calls made, nothing written.")
        return

    with writer.driver() as drv:
        if not writer.check_apoc(drv):
            sys.exit("APOC not found. Run:  python ingest.py setup")

        writer.write_document(drv, document)
        print("  Document and chunks written")

        _banner("Extracting")
        stats = ExtractionStats()
        all_triples = []

        for chunk in document.chunks:
            stats.chunks_seen += 1
            try:
                triples = extract_chunk(chunk.text, chunk.id, stats)
            except Exception as exc:  # keep going; one bad chunk is not fatal
                print(f"  chunk {chunk.seq}: FAILED ({type(exc).__name__}: {exc})")
                continue

            all_triples.extend(triples)
            print(
                f"  chunk {chunk.seq + 1}/{len(document.chunks)}"
                f" -> {len(triples)} triples",
                end="\r",
                flush=True,
            )

        print(" " * 60, end="\r")
        print(f"  {stats.triples_kept} triples kept")

        writer.write_triples(drv, all_triples)
        print("  Written to Neo4j")

    _banner("Summary")
    print(f"  Chunks processed     {stats.chunks_called:>8,}")
    print(f"  Triples returned     {stats.triples_returned:>8,}")
    print(f"  Triples kept         {stats.triples_kept:>8,}")
    if stats.triples_dropped:
        print(f"  Dropped (invalid)    {stats.triples_dropped:>8,}")
    print(f"  Input tokens         {stats.input_tokens:>8,}")
    print(f"  Cached tokens        {stats.cache_read_tokens:>8,}")
    print(f"  Output tokens        {stats.output_tokens:>8,}")
    print(f"  Elapsed              {time.time() - started:>8.1f}s")

    if stats.chunks_called > 1 and stats.cache_read_tokens == 0:
        print(
            "\n  Note: no cache hits. The ontology prompt may be under the\n"
            "  minimum cacheable size for this model."
        )

    print("\nSee it in Neo4j Browser:")
    print("  MATCH (n:__Entity__)-[r]->(m:__Entity__) RETURN n, r, m LIMIT 100")


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry_run = "--dry-run" in sys.argv[1:]

    if not args:
        print(__doc__)
        return

    command = args[0]
    if command == "setup":
        cmd_setup()
    elif command == "stats":
        cmd_stats()
    elif command == "reset":
        cmd_reset()
    else:
        cmd_ingest(command, dry_run)


if __name__ == "__main__":
    main()
