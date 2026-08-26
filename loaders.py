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
