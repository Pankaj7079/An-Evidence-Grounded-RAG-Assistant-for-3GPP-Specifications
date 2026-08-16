# Agent & Developer Operating Manual: 3GPP Telecom Spec Assistant

Welcome to the **3GPP Telecom Spec Assistant** repository. Every AI Agent, Pair Programmer, and Senior Engineer working on this project MUST read and follow the principles, rules, and workflows documented below before making any changes.

---

## 1. Project Mission & Identity

- **Project Name:** 3GPP Telecom Spec Assistant
- **Target Domain:** 5G Core Network Specifications (Telecom / Cloud-Native)
- **Primary Objective:** Build an evidence-grounded RAG assistant that answers queries strictly using retrieved clauses from:
  1. **3GPP TS 23.501 (v17.4.0, Release 17)** — System Architecture for the 5G System
  2. **3GPP TS 23.502 (v16.9.0, Release 16)** — Procedures for the 5G System
- **Core Philosophy:**
  > **Evidence-grounded RAG system with deterministic citation validation and controlled abstention.**
  > *Never claim mathematical zero-hallucination. Prioritize source fidelity, clause-level attribution, and explicit refusal when supporting evidence is insufficient.*

---

## 2. Mandatory Rules for Agents

1. **Phase-by-Phase Execution:**
   - Always complete the current phase, verify it with automated tests/smoke tests, and obtain user confirmation before progressing to the next phase.
   - Do not generate arbitrary modules ahead of their phase.
2. **Strict Grounding & Abstention:**
   - The LLM is an **explainer of retrieved evidence**, NOT a source of telecom facts.
   - If retrieval results do not contain strong supporting context, return the exact abstention response:
     `"I could not find sufficient supporting evidence in the indexed 3GPP documents."`
3. **Citation Completeness:**
   - Every factual sentence or list item must carry exact citations: `[TS XX.XXX, Clause X.X.X, Page YY]`.
   - Never fabricate or guess clause numbers or page numbers.
4. **Environment & Package Management:**
   - Always use `uv` for python execution and package operations (`uv run ...`, `uv sync`, `uv pip ...`).
5. **No Over-Engineering:**
   - Keep the system clean, modular, and maintainable.
   - Avoid unnecessary abstractions (no GraphRAG/Neo4j, no multi-agent swarms, no complex vector router unless explicitly specified in future phases).
   - Use type hints (`typing`, `pydantic`), logging instead of bare `print`, and explicit error handling.

---

## 3. Directory Layout & Module Responsibilities

```
3gpp-telecom-spec-assistant/
├── AGENTS.md                  <-- This master operating manual
├── AGENT_RULES.md             <-- Senior engineering and coding guardrails
├── SYSTEM_PROMPT.md           <-- Grounded system prompt for LLM generation
├── config.py                  <-- Centralized settings and paths
├── pyproject.toml             <-- uv package definitions
├── requirements.txt           <-- Locked dependencies
├── .env.example               <-- Environment variable templates
├── data/
│   ├── raw/                   <-- Official 3GPP PDFs (TS 23.501, TS 23.502)
│   ├── processed/             <-- Clause-aware JSONL chunks
│   └── evaluation/            <-- Curated test questions & benchmark results
├── storage/
│   ├── qdrant/                <-- Local Qdrant vector database storage
│   └── relationships/         <-- Lightweight entity-relationship JSONL files
├── src/
│   ├── models.py              <-- Pydantic schemas (Chunks, Citations, Relations)
│   ├── ingestion.py           <-- PDF text extraction, header/footer cleaner
│   ├── chunking.py            <-- Clause-bounded chunker with enriched headers
│   ├── indexing.py            <-- Dense (all-MiniLM-L6-v2) & Sparse (BM25) indexing
│   ├── retrieval.py           <-- Hybrid search (Dense + BM25 + RRF) & Evidence Gate
│   ├── relationships.py       <-- Lightweight relation store
│   ├── generation.py          <-- Grounded generation (Gemini/Groq at temp=0)
│   ├── validation.py          <-- Deterministic citation & clause/page validator
│   └── evaluation.py          <-- Benchmark runner (Hit rate, citations, abstention)
├── tests/                     <-- Pytest unit & integration test suite
└── app.py                     <-- Streamlit Web Interface
```

---

## 4. Phase Roadmap & Current State

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Repository, Environment (`uv`), Base Configurations, Rule Files | ✅ Completed & Verified |
| **Phase 2** | Clause-Aware PDF Ingestion, Header Cleaning, Semantic Chunking | ✅ Completed & Verified |
| **Phase 3** | Hybrid Indexing (Dense + BM25 in Qdrant) & Evidence Gate | 🔄 Next |
| **Phase 4** | Grounded LLM Generation (Gemini/Groq) | ⏳ Pending |
| **Phase 5** | Deterministic Citation & Evidence Validation | ⏳ Pending |
| **Phase 6** | Streamlit User Interface | ⏳ Pending |
| **Phase 7** | Evaluation Benchmark & Complete Documentation | ⏳ Pending |

---

## 5. Verification Checklist Before Calling Work Done

1. [ ] Code adheres to type hints and passes `uv run pytest tests/`.
2. [ ] All generated answers strictly quote retrieved chunk citations.
3. [ ] Unsupported/out-of-scope questions trigger deterministic abstention.
4. [ ] No API keys or secret credentials committed.
5. [ ] Streamlit UI displays answers, citations, and expandable raw evidence cleanly.
