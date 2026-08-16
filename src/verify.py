"""CLI tool to inspect extracted chunks against raw 3GPP PDFs."""

import argparse
import json
import sys
from pathlib import Path
import pymupdf

from config import settings

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def verify(doc_code: str = "TS 23.501", clause_query: str = "", page_query: int = 0):
    """Compare extracted chunks side-by-side with original PDF pages."""
    pdf_filename = "ts_123501v170400p.pdf" if "501" in doc_code else "ts_123502v160900p.pdf"
    pdf_path = settings.RAW_DATA_DIR / pdf_filename
    jsonl_path = (
        settings.PROCESSED_DATA_DIR / ("ts23501_chunks.jsonl" if "501" in doc_code else "ts23502_chunks.jsonl")
    )

    if not pdf_path.exists() or not jsonl_path.exists():
        print("Missing raw PDF or processed JSONL file. Run ingestion first.")
        return

    # Open PDF and load extracted chunks
    doc = pymupdf.open(str(pdf_path))
    with open(jsonl_path, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    print("=" * 80)
    print(f"VERIFICATION REPORT FOR {doc_code.upper()} ({pdf_filename})")
    print(f"PDF Total Pages: {len(doc)} | Total Extracted Chunks: {len(chunks)}")
    print("=" * 80)

    # Filter matching chunks by clause or page
    matched_chunks = []
    for c in chunks:
        m = c["metadata"]
        if clause_query and clause_query.lower() in m["section_number"].lower():
            matched_chunks.append(c)
        elif page_query and (str(page_query) in m["page_number"] or m["start_page"] == page_query):
            matched_chunks.append(c)

    if not matched_chunks:
        if clause_query or page_query:
            print(f"No chunks matched query: clause='{clause_query}' page={page_query}")
        else:
            matched_chunks = chunks[:3]

    print(f"\nFound {len(matched_chunks)} matching chunk(s):\n")
    for idx, c in enumerate(matched_chunks[:5], 1):
        m = c["metadata"]
        print(f"--- MATCH #{idx} ---")
        print(f"Chunk ID:     {m['chunk_id']}")
        print(f"Hierarchy:    {m['section_hierarchy']}")
        print(f"Clause:       {m['section_number']} - {m['section_title']}")
        print(f"Page Number:  {m['page_number']}")
        print(f"Chunk Length: {len(c['text'])} characters")
        print("\n[PARSED CHUNK TEXT]:")
        print(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))
        print("-" * 80)

    doc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool to inspect chunks against 3GPP PDFs")
    parser.add_argument("--doc", default="TS 23.501", help="Specification: TS 23.501 or TS 23.502")
    parser.add_argument("--clause", default="Foreword", help="Clause number to inspect (e.g. Foreword, 4.2.4, 6.2.1)")
    parser.add_argument("--page", type=int, default=0, help="Page number to inspect")
    args = parser.parse_args()

    verify(doc_code=args.doc, clause_query=args.clause, page_query=args.page)
