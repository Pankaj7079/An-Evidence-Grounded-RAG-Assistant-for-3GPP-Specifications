# Agent Rules

You are implementing an evidence-grounded 3GPP RAG system.

## Product Scope

The indexed corpus contains only:
- 3GPP TS 23.501 (v17.4.0, Release 17)
- 3GPP TS 23.502 (v16.9.0, Release 16)

Do not add extra documents unless explicitly requested.

## Engineering Rules

- Prefer a working simple implementation over unnecessary abstractions.
- Do not add RAPTOR, Agentic RAG, Neo4j, or multi-agent workflows unless explicitly requested.
- Keep ingestion, retrieval, generation, and validation separate.
- Use Python type hints for public functions and data models.
- Use pathlib for file paths.
- Use logging instead of print for application diagnostics.
- Never use bare except blocks.
- Do not silently ignore parsing or indexing errors.
- Never expose API keys.
- Do not fabricate document metadata, page numbers, or evaluation results.
- Preserve document, release, clause, and page metadata.
- Every generated answer must be validated against retrieved source metadata.
- If evidence is insufficient, return the configured abstention response.
- Do not claim zero hallucination.
- Do not rewrite working modules without a specific reason.

## Development Sequence

1. Setup environment and project structure with uv.
2. Ingest 3GPP PDFs with clause-aware extraction.
3. Test chunk metadata and inspect sample chunks.
4. Build local dense + BM25 hybrid retrieval and evidence gate.
5. Test retrieval without an LLM.
6. Add grounded generation.
7. Add deterministic citation validation.
8. Build Streamlit UI.
9. Add evaluation benchmark and tests.
