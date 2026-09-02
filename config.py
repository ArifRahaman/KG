"""Configuration, loaded from .env (or real environment variables)."""
 
import os
import sys
 
from dotenv import load_dotenv
 
load_dotenv()
 
 
def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.startswith("change-me") or value.endswith("..."):
        sys.exit(
            f"\nMissing config: {name}\n"
            f"Copy .env.example to .env and fill it in.\n"
        )
    return value
 
 
# --- Neo4j ---
NEO4J_URI = _require("NEO4J_URI")
NEO4J_USER = _require("NEO4J_USER")
NEO4J_PASSWORD = _require("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j").strip() or "neo4j"
 
# --- Azure OpenAI ---
AZURE_OPENAI_ENDPOINT = _require("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = _require("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_CHATGPT_DEPLOYMENT = _require("AZURE_OPENAI_CHATGPT_DEPLOYMENT")

# Deployment name of your embedding model. text-embedding-3-small is the cheap
# default; 3-large costs ~6x for a modest gain at this chunk size.
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
).strip()

# --- Tuning ---
MAX_CHUNK_CHARS = int(os.getenv("KG_MAX_CHUNK_CHARS", "2000"))
CHUNK_OVERLAP = int(os.getenv("KG_CHUNK_OVERLAP", "200"))
MIN_CHUNK_CHARS = int(os.getenv("KG_MIN_CHUNK_CHARS", "120"))
 
# How many triples to write to Neo4j per transaction.
WRITE_BATCH_SIZE = int(os.getenv("KG_WRITE_BATCH_SIZE", "500"))

# --- Vector search: parent/child retrieval --------------------------------
# Children are embedded and searched; parents are what get returned to the
# LLM. A small child makes a sharp, undiluted vector -- that is the point.
# A 2000-char parent averaged into one vector is blurry; a 250-char child
# that says one thing matches the query that asks for that one thing.

# Character budget per child. Children pack on sentence (or record)
# boundaries up to this, so real sizes land around 150-300.
CHILD_CHUNK_CHARS = int(os.getenv("KG_CHILD_CHUNK_CHARS", "300"))

# Width of the vector index. Neo4j cannot alter this after creation, so
# changing it means dropping the index and re-embedding everything.
#
# 1536 suits both text-embedding-ada-002 (fixed at 1536) and the
# text-embedding-3-* family (which defaults to 1536 for -small). embed.py
# works out on its own whether the deployment accepts a `dimensions`
# argument, so this only ever needs to match what comes back.
EMBEDDING_DIMENSIONS = int(os.getenv("KG_EMBEDDING_DIMENSIONS", "1536"))

# Texts per embeddings API call.
EMBED_BATCH_SIZE = int(os.getenv("KG_EMBED_BATCH_SIZE", "64"))

# How much of the parent to prepend to a child before embedding it. Without
# this, a child like "It opened an office there the following year" embeds
# into near-noise -- nothing in the vector says which company, or when.
# The prefix is embedded but never stored; the child keeps its raw text.
CONTEXT_PREFIX_CHARS = int(os.getenv("KG_CONTEXT_PREFIX_CHARS", "160"))

# Children per Neo4j write. Deliberately small: every row carries a full
# vector, and the HTTP API has to serialise all of it as JSON.
CHILD_WRITE_BATCH_SIZE = int(os.getenv("KG_CHILD_WRITE_BATCH_SIZE", "50"))

# Retrieval fan-out. Adjacent children of one parent tend to match together,
# so ask for far more children than the number of parents actually wanted.
SEARCH_CHILD_K = int(os.getenv("KG_SEARCH_CHILD_K", "30"))
SEARCH_PARENT_K = int(os.getenv("KG_SEARCH_PARENT_K", "5"))

# A multi-part question is split into standalone sub-queries, each searched
# separately. The cap bounds both the LLM planning call and the fan-out.
SEARCH_MAX_SUBQUERIES = int(os.getenv("KG_SEARCH_MAX_SUBQUERIES", "4"))

# Ceiling on the fused result set. The per-query budget is SEARCH_PARENT_K,
# so four sub-queries can surface up to twenty parents; this keeps the
# prompt from growing without bound while still letting a three-part
# question return passages for all three parts.
SEARCH_MAX_PARENTS = int(os.getenv("KG_SEARCH_MAX_PARENTS", "12"))
 