"""
Embeddings for child chunks.

Only children are ever embedded. Parents are what the LLM reads, but a
2000-char parent averaged into one vector is blurry -- it says a little
about everything and not much about anything. A 250-char child that makes
one claim produces a vector that matches the query asking for that claim.

This module also owns the shared Azure OpenAI client, so search.py can
reuse it rather than building a fourth copy.
"""

from __future__ import annotations

import os

from azure.identity import ClientSecretCredential
from openai import AzureOpenAI, BadRequestError

import config


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


# --- Contextualisation -----------------------------------------------------


def contextualize(child_text: str, parent_text: str, source: str) -> str:
    """
    Build the string that actually gets embedded.

    A child in isolation loses its referents. "It opened a second office
    there the following year" has no company, no place and no date in it --
    embedded alone it lands nowhere near a query naming any of them.

    Prepending the source and the head of the parent puts those referents
    back into the vector. The prefix is embedded and then thrown away; the
    child keeps its own raw text for display, and the parent is what the
    LLM ultimately reads.
    """
    head = " ".join(parent_text[: config.CONTEXT_PREFIX_CHARS].split())
    return f"{source} | {head}\n{child_text}"


# --- Embeddings ------------------------------------------------------------


# Only the text-embedding-3-* family accepts a `dimensions` argument; ada-002
# rejects it with a 400. Rather than make the user declare which family their
# deployment belongs to, ask once and remember the answer for the process.
_send_dimensions = True


def _create(batch: list[str]):
    """One embeddings call, negotiating the `dimensions` argument once."""
    global _send_dimensions

    kwargs: dict = {
        "model": config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        "input": batch,
    }
    if _send_dimensions:
        kwargs["dimensions"] = config.EMBEDDING_DIMENSIONS

    try:
        return client.embeddings.create(**kwargs)
    except BadRequestError as exc:
        if not _send_dimensions or "dimensions" not in str(exc):
            raise
        # ada-002 and friends: fixed-width output, parameter not allowed.
        # Drop it permanently and retry. The dimension is still validated
        # against config below, so a genuine mismatch still fails loudly.
        _send_dimensions = False
        kwargs.pop("dimensions")
        return client.embeddings.create(**kwargs)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings, in batches, preserving input order."""
    if not texts:
        return []

    vectors: list[list[float]] = []
    size = config.EMBED_BATCH_SIZE

    for start in range(0, len(texts), size):
        batch = texts[start : start + size]
        response = _create(batch)

        # The API returns an index on every item. Sort by it rather than
        # trusting arrival order -- a silent misalignment here would attach
        # every vector to the wrong child, and nothing downstream would fail
        # loudly enough to notice.
        vectors.extend(
            item.embedding for item in sorted(response.data, key=lambda i: i.index)
        )

    return vectors


def embed_query(question: str) -> list[float]:
    """Embed a single search query. No context prefix -- there is no parent."""
    return embed_texts([question])[0]


def build_child_rows(parents: list[dict]) -> list[dict]:
    """
    Contextualise and embed every child, shaped for writer.write_children.

    Takes [{id, text, source, children: [Child, ...]}, ...] so that it works
    for both freshly loaded documents and chunks read back out of Neo4j
    during a backfill -- the two arrive in different shapes but need exactly
    the same treatment.

    One batched pass over all children, rather than per parent: the API
    round trip dominates, so batching across the whole document matters far
    more than any per-parent structure.
    """
    pairs = [
        (parent, child) for parent in parents for child in parent["children"]
    ]
    if not pairs:
        return []

    vectors = embed_texts(
        [
            contextualize(child.text, parent["text"], parent["source"])
            for parent, child in pairs
        ]
    )

    return [
        {
            "parent_id": parent["id"],
            "id": child.id,
            "text": child.text,
            "seq": child.seq,
            "embedding": vector,
        }
        for (parent, child), vector in zip(pairs, vectors)
    ]
