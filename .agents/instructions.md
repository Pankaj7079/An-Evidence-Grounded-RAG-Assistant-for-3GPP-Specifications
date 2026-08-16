# AI Agent Mandatory Startup Instructions

When any agent session starts or resumes:

## Step 1: Read Project State
1. Read `AGENTS.md` and `config.py` to understand active paths and configuration.
2. Check the Phase Roadmap in `AGENTS.md` to identify the current phase.
3. Review `SYSTEM_PROMPT.md` for generation and citation requirements.

## Step 2: Core Engineering Guardrails
- **Environment**: Always run commands with `uv run <command>`.
- **Modularity**: Never bundle ingestion, retrieval, generation, and validation into a single script. Respect the `src/` modular layout.
- **Source of Truth**: The original PDF text and metadata are the only source of truth.
- **Abstention is a Feature**: An intentional refusal on an out-of-scope question is a success, not a failure.
- **Zero Hallucination is Impossible**: Use the term *"Evidence-grounded RAG with citation validation and abstention"*.

## Step 3: Verification After Every Change
- Never finish a phase without executing an automated smoke test or unit test with `uv run pytest`.
- Always inspect sample data outputs (e.g. 10 sample chunks in ingestion, top-k retrieved chunks in retrieval) before progressing.
