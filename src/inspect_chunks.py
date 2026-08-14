"""Inspection script to verify metadata and structure of processed chunks."""

import json
import sys
from pathlib import Path
from config import settings

# Force stdout encoding to UTF-8
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def inspect_chunks():
    chunks_file = settings.PROCESSED_DATA_DIR / "chunks.jsonl"
    if not chunks_file.exists():
        print("Chunks file does not exist!")
        return

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    print(f"Total chunks processed: {len(chunks)}")
    doc_counts = {}
    for c in chunks:
        doc = c["metadata"]["document_code"]
        doc_counts[doc] = doc_counts.get(doc, 0) + 1

    print(f"Distribution: {doc_counts}")

    indices = [0, 10, 50, 150, 300, 600, 1000, 1200, 1500, min(1800, len(chunks) - 1)]
    for idx in indices:
        if idx >= len(chunks):
            continue
        c = chunks[idx]
        m = c["metadata"]
        print(f"\n=======================================================")
        print(f"Sample #{idx + 1}")
        print(f"Chunk ID:     {m['chunk_id']}")
        print(f"Document:     {m['document_code']} (Rel-{m['release']} v{m['version']})")
        print(f"Clause:       {m['section_number']} - {m['section_title']}")
        print(f"Page Number:  {m['page_number']}")
        print(f"Content Type: {m['content_type']}")
        print(f"Enriched Header Preview:\n{c['enriched_text'][:250]}...")


if __name__ == "__main__":
    inspect_chunks()
