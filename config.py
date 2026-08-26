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
 