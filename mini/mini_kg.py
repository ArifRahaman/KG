"""
mini_kg.py -- a whole knowledge graph in one readable file.

Same idea as the full project, shrunk so you can read it end to end:

    file  ->  chunks  ->  LLM extraction  ->  graph  ->  query

The ONE big change from the real system: the full project stores the graph
in Neo4j (a real graph database that needs Docker/Aura + the APOC plugin).
This version keeps the graph in a plain Python dict and saves it to a JSON
file. Everything else is the same, just smaller:

    * chunking is rule-based and free (no LLM, no cost)
    * the LLM turns each chunk into (subject, predicate, object) triples
    * entities are deduplicated by a deterministic id, so re-running is safe
    * every edge remembers which chunk it came from  (provenance / citations)

Each section below is labelled with the real file it corresponds to, so you
can jump from this toy to the production code once the shape makes sense.

--------------------------------------------------------------------------
Commands
--------------------------------------------------------------------------
    python mini_kg.py build <file>     load -> chunk -> extract -> graph.json
    python mini_kg.py stats            what is in the graph now
    python mini_kg.py show  <name>     one entity + everything touching it
    python mini_kg.py ask   <question> let the LLM answer from the graph facts
    python mini_kg.py export           write graph.html (open it in a browser)

`build` and `ask` call the LLM and so need the Azure OpenAI settings in the
project's .env (the same one the full pipeline uses, one folder up).
`stats`, `show` and `export` only read graph.json -- no keys, no network.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

# The graph is saved next to this script, so the commands find it no matter
# what directory you run them from.
HERE = Path(__file__).resolve().parent
GRAPH_PATH = HERE / "graph.json"
HTML_PATH = HERE / "graph.html"


# ==========================================================================
# 0.  Config + Azure OpenAI client        (mirrors config.py + extract.py)
# ==========================================================================
# We reuse the SAME .env as the full project. It lives one folder up. The
# client is built lazily -- only `build` and `ask` need it, so `stats`,
# `show` and `export` keep working with no keys installed at all.

try:
    from dotenv import load_dotenv

    load_dotenv(HERE.parent / ".env")
except ImportError:
    pass  # fine for the read-only commands; build/ask will complain clearly

CHAT_MODEL = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT", "gpt-4o").strip() or "gpt-4o"


def _client():
    """Build the Azure OpenAI client, API key first, service principal second."""
    try:
        from openai import AzureOpenAI
    except ImportError:
        sys.exit("The 'openai' package is not installed. Run: pip install openai")

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
    if not endpoint:
        sys.exit(
            "Missing AZURE_OPENAI_ENDPOINT.\n"
            "Fill in the .env one folder up (the full project uses the same one)."
        )

    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    if api_key:
        return AzureOpenAI(
            azure_endpoint=endpoint, api_version=api_version, api_key=api_key
        )

    # No key -> fall back to a service principal (tenant/client/secret in .env).
    from azure.identity import ClientSecretCredential

    cred = ClientSecretCredential(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        client_id=os.environ["AZURE_CLIENT_ID"],
        client_secret=os.environ["AZURE_CLIENT_SECRET"],
    )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        api_key="AZURE_AD",  # placeholder; the token provider is used instead
        azure_ad_token_provider=lambda: cred.get_token(
            "https://cognitiveservices.azure.com/.default"
        ).token,
    )


# ==========================================================================
# 1.  Load + chunk the file                        (mirrors loaders.py)
# ==========================================================================
# No LLM here -- splitting text is rule-based and free. We pack whole
# paragraphs up to a size budget instead of cutting every N characters,
# because a chunk that stops in the middle of a sentence produces garbage
# triples, and extraction is only ever as good as its chunks.
#
# The full loaders.py also handles PDF / CSV / JSON and a small "child" layer
# used for vector search. We drop all of that: any text file works here.

MAX_CHUNK_CHARS = 1500


def load_and_chunk(path_str: str) -> list[str]:
    path = Path(path_str).expanduser()
    if not path.exists():
        sys.exit(f"File not found: {path}")

    raw = path.read_text(encoding="utf-8", errors="replace")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]

    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        if not buffer:
            buffer = para
        elif len(buffer) + 2 + len(para) <= MAX_CHUNK_CHARS:
            buffer = f"{buffer}\n\n{para}"
        else:
            chunks.append(buffer)
            buffer = para
    if buffer:
        chunks.append(buffer)

    if not chunks:
        sys.exit(f"No usable text found in {path.name}.")
    return chunks


# ==========================================================================
# 2.  Extract triples from a chunk            (mirrors extract.py + ontology.py)
# ==========================================================================
# This is the only place the LLM is used to *build* the graph. We hand it a
# chunk of text and ask for a list of (subject, predicate, object) facts.
#
# The ontology is "dynamic": we do not give the model a fixed list of allowed
# types. We only nudge it toward consistent naming, and let it invent the
# labels the text calls for. temperature=0 keeps it repeatable.

SYSTEM_PROMPT = """You extract structured facts from text to build a knowledge graph.

Return a JSON object shaped like:
  {"triples": [
     {"subject": "...", "subject_type": "...",
      "predicate": "...",
      "object": "...", "object_type": "..."}
  ]}

Rules:
  - subject_type / object_type: PascalCase (Person, Organization, Product,
    Location, Event, Concept...). Be specific and reuse the same label for
    the same kind of thing.
  - predicate: UPPER_SNAKE_CASE, active voice, one direction
    (FOUNDED, WORKS_AT, ACQUIRED, LOCATED_IN -- not "was founded by").
  - Extract only facts the text actually states. Never guess.
  - Use entity names exactly as written in the text.
  - If there are no facts, return {"triples": []}.
"""


def extract_triples(client, text: str) -> list[dict]:
    """One chunk -> a list of raw triple dicts from the model."""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(content).get("triples", [])
    except json.JSONDecodeError:
        return []  # one malformed chunk is not worth crashing the whole run


# --- Normalisation: how "Acme Corp" and "Acme Corporation" become one node --
# A deterministic id is what makes re-running safe: the same entity always
# hashes to the same id, so we update it instead of creating a duplicate.
# (This is exactly what the real extract.py does.)

_SUFFIXES = r"\b(inc|llc|ltd|limited|corp|corporation|co|plc|gmbh|sa|nv|ag)\b"


def _norm_label(label: str) -> str:
    parts = re.split(r"[\s_]+", (label or "").strip())
    return "".join(p.capitalize() for p in parts if p) or "Entity"


def _norm_predicate(pred: str) -> str:
    pred = (pred or "").strip()
    if not pred:
        return "RELATED_TO"
    pred = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", pred)  # camelCase -> CAMEL_CASE
    pred = re.sub(r"[\s\-]+", "_", pred)
    return pred.upper()


def entity_id(label: str, name: str) -> str:
    """Same name -> same id, on every run and every machine."""
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"[^a-z0-9]+", " ", n).strip()
    n = re.sub(_SUFFIXES, "", n).strip()
    n = re.sub(r"\s+", " ", n)
    return hashlib.sha1(f"{label}|{n}".encode()).hexdigest()[:16]


# ==========================================================================
# 3.  The graph itself             (replaces writer.py + Neo4j entirely)
# ==========================================================================
# In the real system this is a Neo4j database. Here it is three dicts and a
# JSON file. The operations are the same ones Neo4j's MERGE gives you:
# adding an entity that already exists just updates it; adding an edge that
# already exists just appends the new chunk to its provenance list.


class Graph:
    def __init__(self):
        self.nodes: dict[str, dict] = {}   # id -> {id, name, type}
        self.edges: dict[tuple, dict] = {}  # (sid, pred, oid) -> edge record
        self.chunks: dict[str, str] = {}    # chunk_id -> the text it came from

    # --- writes (the "MERGE" half) ---------------------------------------

    def add_chunk(self, chunk_id: str, text: str) -> None:
        self.chunks[chunk_id] = text

    def _add_node(self, label: str, name: str) -> str:
        node_id = entity_id(label, name)
        # First name we saw wins as the display name; that is good enough here.
        self.nodes.setdefault(node_id, {"id": node_id, "name": name, "type": label})
        return node_id

    def add_triple(self, triple: dict, chunk_id: str) -> bool:
        """Add one extracted fact. Returns False if it was unusable."""
        subject = (triple.get("subject") or "").strip()
        obj = (triple.get("object") or "").strip()
        if not subject or not obj:
            return False

        s_type = _norm_label(triple.get("subject_type", ""))
        o_type = _norm_label(triple.get("object_type", ""))
        predicate = _norm_predicate(triple.get("predicate", ""))

        sid = self._add_node(s_type, subject)
        oid = self._add_node(o_type, obj)

        key = (sid, predicate, oid)
        edge = self.edges.get(key)
        if edge is None:
            edge = {"subject_id": sid, "predicate": predicate,
                    "object_id": oid, "chunk_ids": []}
            self.edges[key] = edge
        if chunk_id not in edge["chunk_ids"]:
            edge["chunk_ids"].append(chunk_id)  # provenance, deduplicated
        return True

    # --- persistence (this is our "database") ----------------------------

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "nodes": list(self.nodes.values()),
                    "edges": list(self.edges.values()),
                    "chunks": self.chunks,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Graph":
        if not path.exists():
            sys.exit(
                f"No graph yet at {path.name}.\n"
                "Build one first:  python mini_kg.py build <your-file.txt>"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        g = cls()
        g.nodes = {n["id"]: n for n in data["nodes"]}
        g.edges = {
            (e["subject_id"], e["predicate"], e["object_id"]): e
            for e in data["edges"]
        }
        g.chunks = data.get("chunks", {})
        return g

    # --- small read helpers ----------------------------------------------

    def name(self, node_id: str) -> str:
        node = self.nodes.get(node_id)
        return node["name"] if node else node_id

    def find(self, query: str) -> list[str]:
        """Node ids whose name contains `query` (case-insensitive)."""
        q = query.lower()
        return [nid for nid, n in self.nodes.items() if q in n["name"].lower()]


# ==========================================================================
# 4.  Commands
# ==========================================================================


def cmd_build(path: str) -> None:
    print(f"\nLoading {path}")
    chunks = load_and_chunk(path)
    print(f"  {len(chunks)} chunk(s)")

    client = _client()
    graph = Graph()
    doc_id = hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:12]

    print("\nExtracting facts (one LLM call per chunk)")
    kept = 0
    for i, text in enumerate(chunks):
        chunk_id = f"{doc_id}:{i}"
        graph.add_chunk(chunk_id, text)
        try:
            triples = extract_triples(client, text)
        except Exception as exc:  # one bad chunk should not sink the run
            print(f"  chunk {i + 1}/{len(chunks)}: FAILED ({type(exc).__name__})")
            continue
        for triple in triples:
            if graph.add_triple(triple, chunk_id):
                kept += 1
        print(f"  chunk {i + 1}/{len(chunks)} -> {len(triples)} facts", end="\r")

    print(" " * 60, end="\r")
    graph.save(GRAPH_PATH)
    print(f"  {len(graph.nodes)} entities, {len(graph.edges)} relationships "
          f"({kept} facts kept)")
    print(f"\nSaved to {GRAPH_PATH.name}. Try:")
    print("  python mini_kg.py stats")
    print("  python mini_kg.py export   (then open graph.html)")


def cmd_stats() -> None:
    graph = Graph.load(GRAPH_PATH)
    print("\nGraph")
    print("-----")
    print(f"  Chunks         {len(graph.chunks):>6}")
    print(f"  Entities       {len(graph.nodes):>6}")
    print(f"  Relationships  {len(graph.edges):>6}")

    by_type: dict[str, int] = {}
    for edge in graph.edges.values():
        by_type[edge["predicate"]] = by_type.get(edge["predicate"], 0) + 1
    if by_type:
        print("\nRelationships by type")
        print("---------------------")
        for pred, count in sorted(by_type.items(), key=lambda kv: -kv[1]):
            print(f"  {pred:<22} {count:>4}")


def cmd_show(query: str) -> None:
    """One entity and every edge touching it -- with the sentence behind it."""
    graph = Graph.load(GRAPH_PATH)
    matches = graph.find(query)
    if not matches:
        print(f"\nNo entity matching '{query}'. Try `stats` to see what's there.")
        return

    for nid in matches:
        node = graph.nodes[nid]
        print(f"\n{node['name']}  ({node['type']})")
        print("-" * (len(node["name"]) + len(node["type"]) + 4))

        outgoing = [e for e in graph.edges.values() if e["subject_id"] == nid]
        incoming = [e for e in graph.edges.values() if e["object_id"] == nid]

        # Arrows point relative to THIS entity: -> it is the subject,
        # <- it is the object of someone else's fact.
        for edge in outgoing:
            print(f"  -> {edge['predicate']:<18} {graph.name(edge['object_id'])}")
        for edge in incoming:
            print(f"  <- {edge['predicate']:<18} {graph.name(edge['subject_id'])}")

        if not outgoing and not incoming:
            print("  (no relationships)")

        # Provenance: show where the first fact came from -- the audit trail
        # that lets you trace any edge back to the exact source text.
        example = (outgoing or incoming)
        if example and example[0]["chunk_ids"]:
            snippet = graph.chunks.get(example[0]["chunk_ids"][0], "")
            snippet = " ".join(snippet.split())[:160]
            print(f"\n  source: \"{snippet}...\"")


def cmd_ask(question: str) -> None:
    """Graph-RAG, lite: dump the facts into the prompt and let the LLM answer.

    The real search.py first does vector search to pick only the relevant
    passages. Our graphs are small, so we just hand over every fact. Same
    idea -- the graph supplies structured facts, the LLM turns them into
    prose -- minus the retrieval step.
    """
    graph = Graph.load(GRAPH_PATH)
    facts = "\n".join(
        f"{graph.name(e['subject_id'])} {e['predicate']} {graph.name(e['object_id'])}"
        for e in graph.edges.values()
    )
    if not facts:
        print("\nThe graph is empty. Build one first.")
        return

    client = _client()
    prompt = (
        "Answer the question using ONLY these facts from a knowledge graph.\n"
        "If the facts do not contain the answer, say so plainly.\n\n"
        f"Question: {question}\n\nFacts:\n{facts}\n"
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600,
    )
    print(f"\n{response.choices[0].message.content.strip()}\n")


def cmd_export() -> None:
    """Write a standalone HTML page that draws the graph in the browser."""
    graph = Graph.load(GRAPH_PATH)

    # Colour each node by its type so the shape of the data is visible at a
    # glance. vis-network handles the physics; we just hand it nodes + edges.
    nodes = [
        {"id": n["id"], "label": n["name"], "group": n["type"], "title": n["type"]}
        for n in graph.nodes.values()
    ]
    edges = [
        {"from": e["subject_id"], "to": e["object_id"], "label": e["predicate"]}
        for e in graph.edges.values()
    ]

    HTML_PATH.write_text(
        _HTML_TEMPLATE.replace("__NODES__", json.dumps(nodes, ensure_ascii=False))
        .replace("__EDGES__", json.dumps(edges, ensure_ascii=False)),
        encoding="utf-8",
    )
    print(f"\nWrote {HTML_PATH}")
    print("Open it in a browser. (Rendering pulls vis-network from a CDN,")
    print("so you need to be online the first time.)")


_HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>mini knowledge graph</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    html, body { margin: 0; height: 100%; background: #14161a; }
    #net { width: 100%; height: 100%; }
  </style>
</head>
<body>
  <div id="net"></div>
  <script>
    const nodes = new vis.DataSet(__NODES__);
    const edges = new vis.DataSet(__EDGES__);
    new vis.Network(document.getElementById("net"), {nodes, edges}, {
      nodes: { shape: "dot", size: 16, font: { color: "#e6e6e6" } },
      edges: {
        arrows: "to",
        color: { color: "#5a6472" },
        font: { color: "#9aa4b2", size: 10, strokeWidth: 0 }
      },
      physics: { stabilization: true, barnesHut: { springLength: 140 } }
    });
  </script>
</body>
</html>
"""


# ==========================================================================
# 5.  CLI
# ==========================================================================


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    command, rest = args[0], args[1:]
    if command == "build" and rest:
        cmd_build(rest[0])
    elif command == "stats":
        cmd_stats()
    elif command == "show" and rest:
        cmd_show(" ".join(rest))
    elif command == "ask" and rest:
        cmd_ask(" ".join(rest))
    elif command == "export":
        cmd_export()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
