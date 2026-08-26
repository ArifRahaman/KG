"""
Dynamic ontology: no fixed schema.

The LLM is free to return any node types and relationship types it discovers
in the text. Guidelines in describe() steer the model toward consistent
naming conventions but do NOT restrict what it can extract.
"""


def describe() -> str:
    """Guidelines for the extraction prompt (no fixed types)."""
    return """EXTRACTION GUIDELINES:

NODE TYPES:
  - Use PascalCase labels (e.g. Person, Organization, Drug, Treaty, City)
  - Be specific: prefer "PharmaceuticalCompany" over "Company" when the
    text makes the distinction clear
  - Reuse the same label for the same kind of thing across the document

RELATIONSHIP TYPES:
  - Use UPPER_SNAKE_CASE (e.g. WORKS_AT, APPROVED_BY, LOCATED_IN)
  - Use active voice, one direction only (WORKS_AT, not EMPLOYED_BY)
  - Be specific: prefer MANUFACTURES over RELATED_TO when meaning is clear

GENERAL:
  - Extract only facts the text actually states. Never infer or guess.
  - Use the exact entity name as written in the text.
  - The same entity may appear in multiple triples.
  - If no facts can be extracted, return an empty list."""
