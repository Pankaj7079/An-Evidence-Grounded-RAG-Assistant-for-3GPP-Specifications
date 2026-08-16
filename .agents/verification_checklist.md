# Verification Checklist for Each Phase

## Phase 1 Checklist: Setup & Tooling
- [x] `uv` environment created and all dependencies synced.
- [x] PDF files verified in `data/raw/`.
- [x] Config, rules, and prompt templates in place.
- [x] Smoke test passes without errors.

## Phase 2 Checklist: Ingestion & Chunking
- [x] PyMuPDF extracts text page-by-page accurately.
- [x] Running headers, footers, and copyright noise removed.
- [x] 3GPP clauses correctly extracted with numeric hierarchy.
- [x] Chunk tokens kept within ~350-500 tokens (max 2502 chars = ~600 tokens) without splitting procedure step chains arbitrarily.
- [x] 10 sample chunks inspected and verified for metadata correctness.
- [x] Change-history / CR table noise detected and dropped (28 noise sections removed).
- [x] Bad section numbers (large integers, document-reference patterns like 23.501) filtered.
- [x] Oversized chunks fixed: max 2502 chars, avg 1400 chars (was 7551 chars max).
- [x] Unit tests pass in `tests/test_chunking.py` (3/3 passed).

## Phase 3 Checklist: Hybrid Indexing & Retrieval
- [ ] Local Qdrant collection created and dense embeddings stored.
- [ ] BM25 sparse index persisted and loaded.
- [ ] Hybrid search with RRF returns relevant clauses for technical telecom queries.
- [ ] Evidence gate correctly flags weak queries.
- [ ] Unit tests pass in `tests/test_retrieval.py`.

## Phase 4 & 5 Checklist: Generation & Validation
- [ ] Grounded prompt forces citation generation.
- [ ] Out-of-scope queries trigger exact abstention response.
- [ ] Deterministic validator catches invalid citations.
- [ ] Unit tests pass in `tests/test_validation.py`.

## Phase 6 & 7 Checklist: UI, Evaluation & Docs
- [ ] Streamlit interface starts smoothly and renders evidence cleanly.
- [ ] 30-question evaluation dataset benchmarked.
- [ ] Metrics (Hit rate, citation accuracy, abstention rate) generated.
- [ ] All documentation updated (`README.md`, `ARCHITECTURE.md`, `EVALUATION.md`, `LIMITATIONS.md`).
