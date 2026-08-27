"""
Chunk text -> triples, using Azure OpenAI GPT-4o with dynamic extraction.
 
The model is free to return any node types and relationship types it
discovers in the text. Structured outputs force valid JSON; naming
conventions are guided by the system prompt.
"""
 
from __future__ import annotations
 
import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
 
from azure.identity import ClientSecretCredential
from openai import AzureOpenAI
from pydantic import BaseModel
 
import config
import ontology
 
 
# --- Azure OpenAI client ---------------------------------------------------
 
def _build_client() -> AzureOpenAI:
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    if api_key:
        return AzureOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_version=config.AZURE_OPENAI_API_VERSION,
            api_key=api_key,
        )
 
    # Service-principal auth
    credential = ClientSecretCredential(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        client_id=os.environ["AZURE_CLIENT_ID"],
        client_secret=os.environ["AZURE_CLIENT_SECRET"],
    )
    return AzureOpenAI(
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_version=config.AZURE_OPENAI_API_VERSION,
        api_key="AZURE_AD",  # placeholder; token_provider is used instead
        azure_ad_token_provider=lambda: credential.get_token(
            "https://cognitiveservices.azure.com/.default"
        ).token,
    )
 
 
client = _build_client()
 
 
# --- What we ask the model to return (fully dynamic) ----------------------
 
 
class RawTriple(BaseModel):
    subject: str
    subject_type: str       # any PascalCase label
    predicate: str          # any UPPER_SNAKE_CASE relationship
    object: str
    object_type: str        # any PascalCase label
 
 
class Extraction(BaseModel):
    triples: list[RawTriple]
 
 
# --- What we hand to the writer ------------------------------------------
 
 
@dataclass
class Triple:
    subject_id: str
    subject_name: str
    subject_type: str
    predicate: str
    object_id: str
    object_name: str
    object_type: str
    chunk_id: str
 
 
SYSTEM_PROMPT = f"""You extract structured facts from documents to build a knowledge graph.
 
{ontology.describe()}
 
Return a JSON object with a "triples" array. Each triple has:
  subject, subject_type, predicate, object, object_type
 
Return only the triples."""
 
 
# Strip common company suffixes so "Acme Corp" and "Acme Corporation" collapse
# to the same id.
_SUFFIXES = r"\b(inc|llc|ltd|limited|corp|corporation|co|plc|gmbh|sa|nv|ag)\b"
 
 
def _normalize_label(label: str) -> str:
    """Normalise a dynamic label: strip whitespace, collapse to PascalCase."""
    label = label.strip()
    if not label:
        return "Entity"
    # Already PascalCase? Keep it. Otherwise title-case words and join.
    parts = re.split(r"[\s_]+", label)
    return "".join(p.capitalize() for p in parts if p)
 
 
def _normalize_predicate(pred: str) -> str:
    """Normalise a dynamic predicate to UPPER_SNAKE_CASE."""
    pred = pred.strip()
    if not pred:
        return "RELATED_TO"
    # Insert underscores before uppercase runs (camelCase -> CAMEL_CASE)
    pred = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", pred)
    pred = re.sub(r"[\s\-]+", "_", pred)
    return pred.upper()
 
 
def entity_id(label: str, name: str) -> str:
    """
    Deterministic id: the same entity always gets the same id, on every run
    and every machine.
 
    This is what makes MERGE idempotent. A random uuid here would create a
    fresh duplicate node on every single ingest.
    """
    normalized = unicodedata.normalize("NFKD", name)
    normalized = normalized.encode("ascii", "ignore").decode().lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    normalized = re.sub(_SUFFIXES, "", normalized).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return hashlib.sha1(f"{label}|{normalized}".encode()).hexdigest()[:24]
 
 
@dataclass
class ExtractionStats:
    chunks_seen: int = 0
    chunks_called: int = 0
    triples_returned: int = 0
    triples_kept: int = 0
    triples_dropped: int = 0
    cache_read_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
 
 
def extract_chunk(text: str, chunk_id: str, stats: ExtractionStats) -> list[Triple]:
    """Run one chunk through the model and return triples."""
    stats.chunks_called += 1
 
    response = client.chat.completions.create(
        model=config.AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
 
    usage = response.usage
    if usage:
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            stats.cache_read_tokens += getattr(
                usage.prompt_tokens_details, "cached_tokens", 0
            ) or 0
        stats.input_tokens += usage.prompt_tokens
        stats.output_tokens += usage.completion_tokens
 
    content = response.choices[0].message.content
    if not content:
        return []
 
    try:
        data = json.loads(content)
        parsed = Extraction(**data)
    except Exception:
        return []
 
    kept: list[Triple] = []
    for raw in parsed.triples:
        stats.triples_returned += 1
 
        subject_name = raw.subject.strip()
        object_name = raw.object.strip()
        if not subject_name or not object_name:
            stats.triples_dropped += 1
            continue
 
        subject_type = _normalize_label(raw.subject_type)
        predicate = _normalize_predicate(raw.predicate)
        object_type = _normalize_label(raw.object_type)
 
        kept.append(
            Triple(
                subject_id=entity_id(subject_type, subject_name),
                subject_name=subject_name,
                subject_type=subject_type,
                predicate=predicate,
                object_id=entity_id(object_type, object_name),
                object_name=object_name,
                object_type=object_type,
                chunk_id=chunk_id,
            )
        )
        stats.triples_kept += 1
 
    return kept
 