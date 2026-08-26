"""
FastAPI server for the Knowledge Graph pipeline.
 
    uvicorn api:app --reload
 
Endpoints:
    POST   /ingest           upload a file → background extraction → job_id
    GET    /status/{job_id}  poll job progress
    POST   /setup            create Neo4j constraints & indexes
    POST   /backfill         add the child layer to pre-existing chunks
    GET    /stats            graph summary
    GET    /graph            browse nodes + relationships
    POST   /chat             question → Cypher → answer  (structure)
    POST   /search           question → vector + graph → answer  (text)
    DELETE /reset            wipe all data
"""
 
from __future__ import annotations
 
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional
 
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
 
import embed
import loaders
import search
import writer
from extract import ExtractionStats, extract_chunk
 
app = FastAPI(
    title="Knowledge Graph API",
    description="Upload documents → extract facts with GPT-4o → build a Neo4j knowledge graph",
    version="1.0.0",
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------
 
jobs: dict[str, dict] = {}
 
 
def _run_ingestion(job_id: str, file_path: str, original_name: str) -> None:
    """Background task: load, extract, write."""
    jobs[job_id]["status"] = "running"
 
    try:
        # 1. Load & chunk
        document = loaders.load(file_path)
        jobs[job_id]["chunks"] = len(document.chunks)
        jobs[job_id]["children"] = sum(len(c.children) for c in document.chunks)

        # 2. Write document + chunks
        with writer.driver() as drv:
            if not writer.check_apoc(drv):
                raise RuntimeError(
                    "APOC is not installed. Run POST /setup first, "
                    "or install APOC on your Neo4j instance."
                )

            writer.write_document(drv, document)

            # 2b. Embed children. Without this the upload is invisible to
            # POST /search -- vector search has nothing to match against,
            # because only children carry embeddings.
            rows = embed.build_child_rows(
                [
                    {
                        "id": c.id,
                        "text": c.text,
                        "source": original_name or Path(file_path).name,
                        "children": c.children,
                    }
                    for c in document.chunks
                ]
            )
            writer.write_children(drv, rows)
            jobs[job_id]["children_embedded"] = len(rows)

            # 3. Extract triples
            stats = ExtractionStats()
            all_triples = []
 
            for chunk in document.chunks:
                stats.chunks_seen += 1
                try:
                    triples = extract_chunk(chunk.text, chunk.id, stats)
                except Exception as exc:
                    jobs[job_id].setdefault("errors", []).append(
                        f"chunk {chunk.seq}: {type(exc).__name__}: {exc}"
                    )
                    continue
                all_triples.extend(triples)
 
            # 4. Write triples
            writer.write_triples(drv, all_triples)
 
        jobs[job_id]["status"] = "done"
        jobs[job_id]["stats"] = asdict(stats)
        jobs[job_id]["triples_written"] = len(all_triples)
        jobs[job_id]["finished_at"] = time.time()
 
    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = f"{type(exc).__name__}: {exc}"
 
    finally:
        # Clean up temp file
        try:
            Path(file_path).unlink(missing_ok=True)
        except OSError:
            pass
 
 
# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
 
 
@app.post("/ingest", summary="Upload a file and start extraction")
async def ingest(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a PDF, CSV, JSON, JSONL, TXT, or MD file.
    Returns a job_id immediately; extraction runs in the background.
    Poll GET /status/{job_id} for progress.
    """
    suffix = Path(file.filename or "upload.txt").suffix.lower()
    if suffix not in loaders.SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(loaders.SUPPORTED))}",
        )
 
    # Save upload to a temp file (loaders.load needs a path)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
    except Exception:
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)
        raise
 
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "filename": file.filename,
        "started_at": time.time(),
    }
 
    background_tasks.add_task(_run_ingestion, job_id, tmp.name, file.filename or "")
 
    return {"job_id": job_id, "status": "pending", "message": "Extraction started. Poll GET /status/{job_id}"}
 
 
@app.get("/status/{job_id}", summary="Check job status")
async def status(job_id: str):
    """Poll this endpoint to check if an ingestion job has finished."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]
 
 
@app.post("/setup", summary="Create Neo4j constraints and indexes")
async def setup():
    """Run once to create uniqueness constraints and full-text indexes."""
    with writer.driver() as drv:
        if not writer.check_apoc(drv):
            raise HTTPException(
                status_code=503,
                detail="APOC is not installed on this Neo4j instance.",
            )
        writer.setup(drv)
    return {"message": "Constraints and indexes created."}
 
 
@app.get("/stats", summary="Graph statistics")
async def stats():
    """Return counts of documents, chunks, entities, and relationships by type."""
    with writer.driver() as drv:
        return writer.graph_stats(drv)
 
 
@app.get("/graph", summary="Browse the knowledge graph")
async def graph(
    limit: int = Query(100, ge=1, le=1000, description="Max relationships to return"),
    skip: int = Query(0, ge=0, description="Offset for pagination"),
    node_type: Optional[str] = Query(None, description="Filter by node label, e.g. 'Person'"),
):
    """Return nodes and relationships from the knowledge graph."""
    with writer.driver() as drv:
        if node_type:
            query = (
                f"MATCH (s:__Entity__:`{node_type}`)-[r]->(o:__Entity__) "
                f"RETURN s.id AS source_id, s.name AS source_name, labels(s) AS source_labels, "
                f"type(r) AS relationship, r.chunk_ids AS chunk_ids, "
                f"o.id AS target_id, o.name AS target_name, labels(o) AS target_labels "
                f"SKIP $skip LIMIT $limit"
            )
        else:
            query = (
                "MATCH (s:__Entity__)-[r]->(o:__Entity__) "
                "RETURN s.id AS source_id, s.name AS source_name, labels(s) AS source_labels, "
                "type(r) AS relationship, r.chunk_ids AS chunk_ids, "
                "o.id AS target_id, o.name AS target_name, labels(o) AS target_labels "
                "SKIP $skip LIMIT $limit"
            )
 
        results = drv.run(query, {"skip": skip, "limit": limit})
        edges = []
        nodes_seen: dict[str, dict] = {}
 
        for data in results:
            src_id = data["source_id"]
            tgt_id = data["target_id"]
            if src_id not in nodes_seen:
                nodes_seen[src_id] = {
                    "id": src_id,
                    "name": data["source_name"],
                    "labels": [l for l in data["source_labels"] if l != "__Entity__"],
                }
            if tgt_id not in nodes_seen:
                nodes_seen[tgt_id] = {
                    "id": tgt_id,
                    "name": data["target_name"],
                    "labels": [l for l in data["target_labels"] if l != "__Entity__"],
                }
 
            edges.append({
                "source": src_id,
                "target": tgt_id,
                "relationship": data["relationship"],
                "chunk_ids": data.get("chunk_ids", []),
            })
 
    return {
        "nodes": list(nodes_seen.values()),
        "edges": edges,
        "count": len(edges),
    }
 
 
@app.delete("/reset", summary="Wipe all graph data")
async def reset():
    """Delete all Documents, Chunks, and Entities."""
    with writer.driver() as drv:
        deleted = writer.wipe(drv)
    return {"message": f"Graph cleared. {deleted:,} nodes deleted."}
 
 
# ---------------------------------------------------------------------------
# Chat: natural-language question → Cypher → answer
# ---------------------------------------------------------------------------
 
from chat import text_to_cypher, answer_question, _get_schema, _find_matching_entities
from pydantic import BaseModel
 
 
class ChatRequest(BaseModel):
    question: str
 
 
@app.post("/chat", summary="Ask a question about the knowledge graph")
async def chat(body: ChatRequest):
    """
    Send `{"question": "..."}` and get back a natural-language answer
    derived from a Cypher query over the knowledge graph.
    """
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Missing 'question' field")
 
    with writer.driver() as drv:
        schema = _get_schema(drv)
        entity_matches = _find_matching_entities(drv, question)
        cypher = text_to_cypher(question, schema, entity_matches)
 
        # Try up to 2 times: retry on error or empty results
        for attempt in range(2):
            try:
                results = drv.run(cypher)
            except Exception as exc:
                if attempt == 0:
                    cypher = text_to_cypher(
                        question, schema, entity_matches,
                        f"The previous query failed with: {exc}\nGenerate a simpler, corrected query.",
                    )
                    continue
                return {"question": question, "cypher": cypher, "error": str(exc), "answer": None}
 
            if not results and attempt == 0:
                cypher = text_to_cypher(
                    question, schema, entity_matches,
                    "The previous query returned no results. Try a broader pattern: "
                    "use undirected relationships (a)-[r]-(b), or remove label filters, "
                    "or use CONTAINS for name matching.",
                )
                continue
            break
 
        answer = answer_question(question, results)
 
    return {"question": question, "cypher": cypher, "results": results[:50], "answer": answer}


# ---------------------------------------------------------------------------
# Search: vector retrieval over chunk text, enriched with graph facts
# ---------------------------------------------------------------------------
#
# /chat and /search answer different questions and neither replaces the other.
#
#   /chat    turns the question into Cypher. Precise on structure -- "who is
#            the CEO", "what did X acquire" -- and blind to anything the
#            graph has no edge for.
#   /search  matches the question against the source text. Finds reasons,
#            caveats and qualifiers that extraction discarded, because a
#            triple keeps that X acquired Y but never why.
#
# Prefer /chat for "who/what/which", /search for "why/how/explain".


class SearchRequest(BaseModel):
    question: str
    k_child: Optional[int] = None   # children to fetch before collapsing
    k_parent: Optional[int] = None  # parent chunks to return
    include_facts: bool = True


@app.post("/search", summary="Ask a question answered from the source text")
async def search_endpoint(body: SearchRequest):
    """
    Send `{"question": "..."}` and get back an answer grounded in the
    retrieved passages, plus the passages themselves so the caller can
    show its own citations.

    Retrieval runs against ChildChunk vectors and then hops to the parent
    Chunk, so matching stays precise while the model still reads whole
    passages.
    """
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Missing 'question' field")

    with writer.driver() as drv:
        try:
            passages = search.retrieve(
                drv, question, k_child=body.k_child or 0, k_parent=body.k_parent or 0
            )
        except Exception as exc:
            # Overwhelmingly the missing vector index rather than a bad query.
            raise HTTPException(
                503,
                f"Vector search failed ({exc}). Has POST /setup been run, "
                "and is anything ingested with embeddings?",
            )

        if not passages:
            return {
                "question": question,
                "answer": None,
                "passages": [],
                "facts": [],
                "message": "Nothing retrieved. Check GET /stats for embedded children.",
            }

        facts = (
            search.expand(drv, [p["chunk_id"] for p in passages])
            if body.include_facts
            else []
        )
        answer = search.answer(question, passages, facts)

    return {
        "question": question,
        "answer": answer,
        "passages": [
            {
                "chunk_id": p["chunk_id"],
                "source": p["source"],
                "text": p["text"],
                "score": p["score"],
                "matched_children": p["matched_children"],
            }
            for p in passages
        ],
        "facts": facts,
    }


@app.post("/backfill", summary="Add the child layer to pre-existing chunks")
async def backfill():
    """
    Embed children for chunks ingested before the vector layer existed.

    Rebuilds them from the chunk text already stored in Neo4j, so the
    original source file does not need to still be around.
    """
    with writer.driver() as drv:
        pending = writer.chunks_without_children(drv)
        if not pending:
            return {"message": "Every chunk already has children.", "children": 0}

        rows = embed.build_child_rows(
            [
                {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "source": Path(chunk["source"] or "unknown").name,
                    "children": loaders._make_children(chunk["id"], chunk["text"]),
                }
                for chunk in pending
            ]
        )
        writer.write_children(drv, rows)

    return {
        "message": f"Backfilled {len(pending)} chunk(s).",
        "chunks": len(pending),
        "children": len(rows),
    }
 