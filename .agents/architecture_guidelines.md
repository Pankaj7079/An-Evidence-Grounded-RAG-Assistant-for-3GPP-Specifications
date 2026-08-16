# Architecture Guidelines

## 1. Data Ingestion & Chunking
- **Parser**: `pymupdf` (fitz).
- **Header/Footer Cleaner**: Strip repeated running headers ("3GPP TS 23.501 V17.4.0"), ETSI copyright lines, standalone page number artifacts from text content.
- **Page Numbers**: Preserved strictly in chunk metadata (`page_number`).
- **Clause Detection**: Match 3GPP numbered headings (e.g. `^(\d+(\.\d+)+)\s+([A-Z].*)$`).
- **Enriched Chunk Header**:
  ```text
  Document: 3GPP TS 23.501
  Title: System architecture for the 5G System
  Release: 17
  Clause: 5.2.2
  Clause title: Access and Mobility Management Function
  Page: 42
  Content:
  <Extracted text>
  ```

## 2. Hybrid Retrieval Strategy
- **Dense Vector Search**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) in local Qdrant.
- **Sparse Lexical Search**: `rank_bm25` for exact acronyms (AMF, SMF, UPF, N2, N4, etc.).
- **Fusion**: Reciprocal Rank Fusion (RRF with `k=60`).
- **Evidence Gate**: Check top scores and term matches. If no relevant chunks match, trigger abstention early.

## 3. Grounded Generation & Validation
- **LLM Settings**: Temperature = 0.0, strict system prompt from `SYSTEM_PROMPT.md`.
- **Validation**: Regex extraction of all `[TS XX.XXX, Clause X.X.X, Page YY]` citations. Compare against retrieved chunks. If citations are fabricated or outside retrieved context, reject or trigger abstention.
