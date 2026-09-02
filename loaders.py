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
class Child:
    """
    A retrieval unit. Small on purpose.

    Children are the only things that get embedded. They exist to be found,
    never to be read -- once a search lands on one, the pipeline hops to its
    parent Chunk and hands *that* to the LLM.
    """

    id: str
    text: str
    seq: int


@dataclass
class Chunk:
    id: str
    text: str
    seq: int
    content_hash: str
    children: list[Child] = field(default_factory=list)


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
        chunk_id = f"{doc_id}:{seq}"
        chunks.append(
            Chunk(
                id=chunk_id,
                text=text,
                seq=seq,
                content_hash=content_hash,
                children=_make_children(chunk_id, text),
            )
        )
        seq += 1
    return chunks


# --------------------------------------------------------------------------
# Child splitting -- the retrieval layer
# --------------------------------------------------------------------------


def _split_units(text: str) -> list[str]:
    """
    Break a chunk into its smallest natural pieces.

    Records first: CSV and JSON chunks are already several complete records
    joined by RECORD_SEPARATOR, and one record is a far better retrieval unit
    than a sentence cut out of the middle of one.

    Otherwise sentences, because a child that ends mid-clause embeds badly.
    """
    if RECORD_SEPARATOR in text:
        return [u.strip() for u in text.split(RECORD_SEPARATOR) if u.strip()]
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _hard_split(text: str, budget: int) -> list[str]:
    """
    Last resort for a single unit longer than the budget.

    Splits on whitespace, never mid-word -- a vector for '...ed Acme in 20'
    is worse than no vector at all.
    """
    out: list[str] = []
    buffer = ""
    for word in text.split():
        if not buffer:
            buffer = word
        elif len(buffer) + 1 + len(word) <= budget:
            buffer = f"{buffer} {word}"
        else:
            out.append(buffer)
            buffer = word
    if buffer:
        out.append(buffer)
    return out


def _make_children(chunk_id: str, text: str) -> list[Child]:
    """Pack a chunk's units into children of at most CHILD_CHUNK_CHARS."""
    budget = config.CHILD_CHUNK_CHARS
    packed: list[str] = []
    buffer = ""

    for unit in _split_units(text):
        if len(unit) > budget:
            if buffer:
                packed.append(buffer)
                buffer = ""
            packed.extend(_hard_split(unit, budget))
            continue

        if not buffer:
            buffer = unit
        elif len(buffer) + 1 + len(unit) <= budget:
            buffer = f"{buffer} {unit}"
        else:
            packed.append(buffer)
            buffer = unit

    if buffer:
        packed.append(buffer)

    return [
        Child(id=f"{chunk_id}#{i}", text=t, seq=i) for i, t in enumerate(packed)
    ]
 
 
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
 
 
# --------------------------------------------------------------------------
# PDF text repair -- runs before any chunking
# --------------------------------------------------------------------------

# pypdf hands back glyphs as typeset, not as written. Justified academic text
# is hyphenated at the right margin, so "infrastructure flexibility" arrives
# as "infrastruc-\nture flexibility" and matches no query asking for it.
# Ligatures and typographic dashes arrive as codepoints no query contains
# either.
_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl",
    "\u2010": "-", "\u2011": "-", "\u00ad": "-",
    "\u00a0": " ", "\ufffd": "",
}

# A hyphen with letters on both sides *on one line* is a real compound
# ("multi-actor"). Neither alternative can match across a newline, so every
# hit here is same-line by construction.
_INLINE_COMPOUND = re.compile(r"\b([A-Za-z]{2,})-([A-Za-z]{2,})\b")
_HYPHEN_BREAK = re.compile(r"\b([A-Za-z]{2,})-\n([a-z]{2,})\b")

# Fallback for compounds the corpus only ever shows broken, so it never
# witnesses them intact.
_PREFIXES = {
    "non", "multi", "self", "pre", "post", "co", "re", "sub", "cross",
    "inter", "intra", "semi", "anti", "meta", "socio", "techno",
}


def _dehyphenate(text: str) -> str:
    """
    Rejoin words split across a line break, keeping genuine compounds.

    The document is its own dictionary: a pair seen hyphenated on a single
    line somewhere keeps its hyphen everywhere. That beats a fixed wordlist,
    which would have to know every domain term in advance.
    """
    compounds = {
        (a.lower(), b.lower()) for a, b in _INLINE_COMPOUND.findall(text)
    }

    def join(match: re.Match) -> str:
        head, tail = match.group(1), match.group(2)
        keep = (
            (head.lower(), tail.lower()) in compounds
            or head.lower() in _PREFIXES
        )
        return f"{head}-{tail}" if keep else f"{head}{tail}"

    return _HYPHEN_BREAK.sub(join, text)


def _repair_pdf_text(text: str) -> str:
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    return _dehyphenate(text)


# --------------------------------------------------------------------------
# Section detection -- the structural unit for PDFs
# --------------------------------------------------------------------------

# "4.3 From Cloud Experimentation to Sovereign and Secure Operational
#  Infrastructure (T-3)" -- numbered, title-cased, and often wrapped across
# two lines by the typesetter.
_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\s+([A-Z].{2,90})$")


def _looks_like_heading(line: str) -> bool:
    # A comma means prose or an address block ("1 University of Hamburg,
    # Hamburg, Germany") rather than a section title.
    return (
        bool(_HEADING.match(line))
        and not line.endswith(".")
        and "," not in line
    )


def _split_sections(text: str) -> list[tuple[str, str]]:
    """
    Cut the document at its section headings, returning (heading, body).

    Without this, chunk boundaries fall wherever the character budget runs
    out, and an identifier that lives in a heading ends up in a different
    chunk from the paragraphs explaining it.
    """
    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = [("", [])]
    skip_next = False

    for i, raw in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        line = raw.strip()
        if _looks_like_heading(line):
            heading = line
            # Absorb a wrapped continuation: short, unpunctuated, capitalised.
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if nxt and len(nxt) < 60 and not nxt.endswith(".") and nxt[:1].isupper():
                heading = f"{heading} {nxt}"
                skip_next = True
            sections.append((heading, []))
            continue

        sections[-1][1].append(line)

    return [(h, " ".join(b).strip()) for h, b in sections if " ".join(b).strip()]


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

    repaired = _repair_pdf_text("\n".join(pages))

    # Every chunk opens with its heading. That costs one line and buys three
    # things: the heading is embedded into each child through the context
    # prefix, it survives into what the LLM reads, and an identifier that
    # appears only in a heading still anchors the body beneath it.
    texts: list[str] = []
    for heading, body in _split_sections(repaired):
        label = f"[{heading}]\n\n" if heading else ""
        for piece in _split_long(body):
            texts.append(f"{label}{piece}")
    return texts


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
 