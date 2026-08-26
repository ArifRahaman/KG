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
 
 
...
 
"""
Turn a file into a Document plus a list of Chunks.
 
No LLM is used here. Chunking is rule-based and free.
 
Supported:
  .pdf                -> text per page, then packed into chunks
  .txt .md            -> paragraph split, then packed
  .csv .tsv           -> ONE CHUNK PER ROW (rows are already records)
  .json .jsonl        -> one chunk per record
"""
 
from __future__ import annotations
 
import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
 
import config
 
SUPPORTED = {".pdf", ".txt", ".md", ".csv", ".tsv", ".json", ".jsonl"}
 
 
@dataclass
class Chunk:
    id: str
    text: str
    seq: int
    content_hash: str
 
 
@dataclass
class Document:
    id: str
    source_uri: str
    content_hash: str
    chunks: list[Chunk] = field(default_factory=list)
 
 
def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
 
 
def _doc_id(path: Path) -> str:
    return _sha(str(path.resolve()))[:24]
 
 
def _make_chunks(doc_id: str, texts: list[str], min_chars: int) -> list[Chunk]:
    """Wrap raw text pieces as Chunks, dropping ones too short to be useful."""
    chunks: list[Chunk] = []
    seq = 0
    for text in texts:
        text = text.strip()
        if len(text) < min_chars:
            continue
        content_hash = _sha(text)
        chunks.append(
            Chunk(
                id=f"{doc_id}:{seq}",
                text=text,
                seq=seq,
                content_hash=content_hash,
            )
        )
        seq += 1
    return chunks
 
 
def _pack_paragraphs(text: str) -> list[str]:
    """
    Split on blank lines, then pack paragraphs up to MAX_CHUNK_CHARS.
 
    Structure-aware beats fixed-size: a chunk that straddles two sections
    produces garbage triples, and extraction quality is capped by chunk
    quality.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    packed: list[str] = []
    buffer = ""
 
    for para in paragraphs:
        # A single oversized paragraph gets split on sentence boundaries.
        if len(para) > config.MAX_CHUNK_CHARS:
            if buffer:
                packed.append(buffer)
                buffer = ""
            packed.extend(_split_long(para))
            continue
 
        if not buffer:
            buffer = para
        elif len(buffer) + 2 + len(para) <= config.MAX_CHUNK_CHARS:
            buffer = f"{buffer}\n\n{para}"
        else:
            packed.append(buffer)
            tail = buffer[-config.CHUNK_OVERLAP :] if config.CHUNK_OVERLAP else ""
            buffer = f"{tail}\n\n{para}".strip() if tail else para
 
    if buffer:
        packed.append(buffer)
    return packed
 
 
def _split_long(text: str) -> list[str]:
    """Break an oversized paragraph on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    buffer = ""
    for sentence in sentences:
        if not buffer:
            buffer = sentence
        elif len(buffer) + 1 + len(sentence) <= config.MAX_CHUNK_CHARS:
            buffer = f"{buffer} {sentence}"
        else:
            out.append(buffer)
            buffer = sentence
    if buffer:
        out.append(buffer)
    return out
 
 
# --------------------------------------------------------------------------
# Per-format readers
# --------------------------------------------------------------------------
 
 
def _read_pdf(path: Path) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise SystemExit("PDF support needs pypdf:  pip install pypdf")
 
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
 
    if not pages:
        raise SystemExit(
            f"No text found in {path.name}.\n"
            "If this is a scanned PDF it needs OCR first -- pypdf only reads "
            "embedded text, not images of text."
        )
 
    # Pack across the whole document so chunks don't stop dead at page breaks.
    return _pack_paragraphs("\n\n".join(pages))
 
 
def _read_text(path: Path) -> list[str]:
    return _pack_paragraphs(path.read_text(encoding="utf-8", errors="replace"))
 
 
RECORD_SEPARATOR = "\n---\n"
 
 
def _pack_records(records: list[str]) -> list[str]:
    """
    Pack whole records into chunks, never splitting one across a boundary.
 
    One chunk per row would mean one LLM call per row -- ruinous on a large
    CSV. Packing several complete records into each chunk cuts the call count
    by 10-30x while keeping every record intact and readable.
    """
    packed: list[str] = []
    buffer: list[str] = []
    size = 0
 
    for record in records:
        record = record.strip()
        if not record:
            continue
        addition = len(record) + len(RECORD_SEPARATOR)
        if buffer and size + addition > config.MAX_CHUNK_CHARS:
            packed.append(RECORD_SEPARATOR.join(buffer))
            buffer, size = [], 0
        buffer.append(record)
        size += addition
 
    if buffer:
        packed.append(RECORD_SEPARATOR.join(buffer))
    return packed
 
 
def _read_csv(path: Path) -> list[str]:
    """Rows are already complete records -- pack them, don't re-chunk them."""
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    rows: list[str] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for record in reader:
            rows.append(_record_to_text(record))
    return _pack_records(rows)
 
 
def _read_json(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
 
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        data = json.loads(raw)
        records = data if isinstance(data, list) else [data]
 
    return _pack_records([_record_to_text(r) for r in records])
 
 
def _record_to_text(record: object) -> str:
    """
    Render a structured record as readable prose-ish text.
 
    The LLM reads 'Name: Ali Rahman' far more reliably than raw JSON, and
    keeping one record per chunk preserves the row boundary.
    """
    if not isinstance(record, dict):
        return str(record)
 
    lines: list[str] = []
    for key, value in record.items():
        if value is None or value == "":
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}: {value}")
    return "\n".join(lines)
 
 
# --------------------------------------------------------------------------
 
 
def load(path_str: str) -> Document:
    path = Path(path_str).expanduser()
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
 
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise SystemExit(
            f"Unsupported file type '{suffix}'.\n"
            f"Supported: {', '.join(sorted(SUPPORTED))}"
        )
 
    # Prose gets a minimum length so page numbers, headers and stray fragments
    # are dropped. Structured records are complete regardless of length, so
    # they only need to be non-trivial.
    if suffix == ".pdf":
        texts = _read_pdf(path)
        min_chars = config.MIN_CHUNK_CHARS
    elif suffix in {".txt", ".md"}:
        texts = _read_text(path)
        min_chars = config.MIN_CHUNK_CHARS
    elif suffix in {".csv", ".tsv"}:
        texts = _read_csv(path)
        min_chars = 10
    else:
        texts = _read_json(path)
        min_chars = 10
 
    doc_id = _doc_id(path)
    chunks = _make_chunks(doc_id, texts, min_chars)
 
    if not chunks:
        raise SystemExit(f"No usable text extracted from {path.name}.")
 
    return Document(
        id=doc_id,
        source_uri=str(path.resolve()),
        content_hash=_sha("".join(c.content_hash for c in chunks)),
        chunks=chunks,
    )
 
 