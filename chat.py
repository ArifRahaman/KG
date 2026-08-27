"""
Natural-language chat over the knowledge graph.
 
    python chat.py
 
Ask a question in plain English. GPT-4o converts it to Cypher,
runs it against Neo4j, and answers in natural language.
"""
 
from __future__ import annotations
 
import os
import sys
 
from azure.identity import ClientSecretCredential
from openai import AzureOpenAI
 
import config
import writer
 
# --- Azure OpenAI client ---------------------------------------------------
 
def _build_client() -> AzureOpenAI:
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    if api_key:
        return AzureOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_version=config.AZURE_OPENAI_API_VERSION,
            api_key=api_key,
        )
    credential = ClientSecretCredential(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        client_id=os.environ["AZURE_CLIENT_ID"],
        client_secret=os.environ["AZURE_CLIENT_SECRET"],
    )
    return AzureOpenAI(
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_version=config.AZURE_OPENAI_API_VERSION,
        api_key="AZURE_AD",
        azure_ad_token_provider=lambda: credential.get_token(
            "https://cognitiveservices.azure.com/.default"
        ).token,
    )
 
llm = _build_client()
 
# --- Schema discovery -------------------------------------------------------
 
def _get_schema(drv: writer.Neo4jHTTP) -> str:
    """Fetch node labels, relationship types, property keys, and sample data."""
    labels = [r["label"] for r in drv.run("CALL db.labels() YIELD label RETURN label")]
    rels = [r["relationshipType"] for r in drv.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")]
    props = [r["propertyKey"] for r in drv.run("CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey")]
 
    # Sample relationships to show actual graph patterns
    samples = drv.run(
        "MATCH (a:__Entity__)-[r]->(b:__Entity__) "
        "RETURN a.name AS src, type(r) AS rel, b.name AS tgt LIMIT 30"
    )
    sample_lines = "\n".join(f"  ({s['src']})-[:{s['rel']}]->({s['tgt']})" for s in samples)
 
    return (
        f"Node labels: {', '.join(labels)}\n"
        f"Relationship types: {', '.join(rels)}\n"
        f"Property keys: {', '.join(props)}\n\n"
        f"Sample relationships in the graph:\n{sample_lines}"
    )
 
 
def _find_matching_entities(drv: writer.Neo4jHTTP, question: str) -> str:
    """Fuzzy-search the graph for entities mentioned in the question,
    then fetch their actual neighborhoods so the LLM sees real connections."""
    stop = {"the","and","for","are","was","were","what","who","which","how",
            "does","did","has","have","been","that","this","with","from","they"}
    words = [w for w in question.split() if len(w) >= 3 and w.lower() not in stop]
 
    if not words:
        return ""
 
    # Find matching entity names
    entity_names = set()
    entity_lines = []
    for word in words:
        clean = word.strip("?,.'\"!").lower()
        if not clean:
            continue
        rows = drv.run(
            "MATCH (e:__Entity__) WHERE toLower(e.name) CONTAINS $term "
            "RETURN e.name AS name, labels(e) AS labels LIMIT 5",
            {"term": clean},
        )
        for r in rows:
            if r["name"] not in entity_names:
                entity_names.add(r["name"])
                type_labels = [l for l in r["labels"] if l != "__Entity__"]
                entity_lines.append(f"  {r['name']} (:{':'.join(type_labels)})")
 
    if not entity_names:
        return ""
 
    # Fetch actual relationships for matched entities (the key improvement)
    neighborhood_lines = []
    for name in list(entity_names)[:10]:
        rels = drv.run(
            "MATCH (a:__Entity__ {name: $name})-[r]-(b:__Entity__) "
            "RETURN a.name AS src, type(r) AS rel, b.name AS tgt, "
            "startNode(r) = a AS outgoing LIMIT 15",
            {"name": name},
        )
        for row in rels:
            if row.get("outgoing", True):
                neighborhood_lines.append(f"  ({row['src']})-[:{row['rel']}]->({row['tgt']})")
            else:
                neighborhood_lines.append(f"  ({row['tgt']})-[:{row['rel']}]->({row['src']})")
 
    parts = ["Entities matching the question:"]
    parts.extend(entity_lines[:20])
    if neighborhood_lines:
        seen = set()
        unique_rels = [l for l in neighborhood_lines if not (l in seen or seen.add(l))]
        parts.append("\nActual relationships of these entities:")
        parts.extend(unique_rels[:30])
    return "\n".join(parts)
 
# --- Text-to-Cypher --------------------------------------------------------
 
SYSTEM_PROMPT = """\
You are a Neo4j Cypher expert. Given a user question, the graph schema, and
matching entities found in the graph, write a single read-only Cypher query.
 
Graph structure:
- All entity nodes have the label __Entity__ plus a type label (e.g. :__Entity__:Person).
- Entity node properties: id (string), name (string), created_at (datetime).
- Relationships connect entity nodes directly: (entity)-[:REL_TYPE]->(entity).
- Relationships may have a chunk_ids list property. They do NOT have date/year/amount properties.
- Information like dates, amounts, or locations are stored as separate entity nodes
  connected by relationships, NOT as properties on relationships.
 
Rules:
- Return ONLY the Cypher query, no explanation, no markdown fences.
- Use the exact node names from "Entities matching the question" when available.
- Don't filter by type label unless you're sure — just use :__Entity__.
- When asked "who does X" or "what is the Y of Z", search broadly using variable-length paths
  or match in both directions: (a)-[r]-(b) to find indirect connections.
- Keep queries simple. Prefer broad patterns over specific ones.
- Always LIMIT results (default 25).
- Never write data (no CREATE, MERGE, SET, DELETE).
 
Examples:
Q: Who is the CEO of TechCorp?
MATCH (company:__Entity__ {{name: "TechCorp Inc."}})<-[r1]-(person:__Entity__)-[r2]->(role:__Entity__)
WHERE toLower(role.name) CONTAINS "ceo"
RETURN person.name AS name, role.name AS role, type(r1) AS relation_to_company
LIMIT 25
 
Q: Board of Directors names?
MATCH (person:__Entity__)-[:MEMBER_OF|CHAIR_OF]->(board:__Entity__)
WHERE toLower(board.name) CONTAINS "board"
RETURN person.name AS name, type(r) AS role
LIMIT 25
 
Q: When was TechCorp founded?
MATCH (company:__Entity__ {{name: "TechCorp Inc."}})-[:FOUNDED_IN]->(date:__Entity__)
RETURN date.name AS founded_year
LIMIT 25
 
Q: Tell me everything about TechCorp
MATCH (n:__Entity__ {{name: "TechCorp Inc."}})-[r]-(m:__Entity__)
RETURN n.name AS entity, type(r) AS relationship, m.name AS related_to
LIMIT 25
 
Schema:
{schema}
 
{entity_matches}
"""
 
ANSWER_PROMPT = """\
The user asked: {question}
 
The Cypher query returned these results:
{results}
 
Answer strictly based on the results above. Do not add information that is not in the results.
If the results are empty, say the information was not found in the knowledge graph.
"""
 
 
def text_to_cypher(question: str, schema: str, entity_matches: str = "", error_context: str = "") -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(schema=schema, entity_matches=entity_matches)},
        {"role": "user", "content": question},
    ]
    if error_context:
        messages.append({"role": "user", "content": error_context})
    resp = llm.chat.completions.create(
        model=config.AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        messages=messages,
        temperature=0,
        max_tokens=512,
    )
    cypher = resp.choices[0].message.content.strip()
    # Strip markdown fences if the model adds them anyway
    if cypher.startswith("```"):
        cypher = "\n".join(cypher.split("\n")[1:])
        if cypher.endswith("```"):
            cypher = cypher[:-3].strip()
    return cypher
 
 
def answer_question(question: str, results: list[dict]) -> str:
    results_str = "\n".join(str(r) for r in results[:50]) if results else "(empty)"
    resp = llm.chat.completions.create(
        model=config.AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        messages=[
            {"role": "user", "content": ANSWER_PROMPT.format(question=question, results=results_str)},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return resp.choices[0].message.content.strip()
 
 
# --- Main loop --------------------------------------------------------------
 
def main() -> None:
    with writer.driver() as drv:
        schema = _get_schema(drv)
        print("Knowledge Graph Chat  (type 'quit' to exit)\n")
 
        if not schema.split("\n")[0].endswith("(none)"):
            print(f"Schema:\n{schema}\n")
        else:
            print("Warning: graph appears empty. Ingest documents first.\n")
 
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
 
            if not question or question.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break
 
            # 1. Find matching entities and generate Cypher
            entity_matches = _find_matching_entities(drv, question)
            cypher = text_to_cypher(question, schema, entity_matches)
            print(f"\nCypher: {cypher}\n")
 
            # 2. Run query
            try:
                results = drv.run(cypher)
            except Exception as exc:
                print(f"Query error: {exc}\n")
                continue
 
            # 3. Answer in natural language
            answer = answer_question(question, results)
            print(f"Answer: {answer}\n")
 
 
if __name__ == "__main__":
    main()
 