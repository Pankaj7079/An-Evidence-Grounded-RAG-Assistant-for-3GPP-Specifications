"""Unit tests for PDF ingestion and authoritative ground-truth chunking."""

import json
from pathlib import Path
import pytest
from config import settings
from src.ingestion import clean_page_lines, parse_toc_entry
from src.models import Chunk, ChunkMetadata


def test_clean_page_lines():
    sample_text = (
        "ETSI\n"
        "ETSI TS 123 501 V17.4.0 (2022-05)\n"
        "42\n"
        "3GPP TS 23.501 version 17.4.0 Release 17\n"
        "5.2.2 Access and Mobility Management Function (AMF)\n"
        "The AMF includes the following functionality: Termination of RAN CP interface (N2)."
    )
    cleaned = clean_page_lines(sample_text)
    assert "ETSI TS 123 501 V17.4.0 (2022-05)" not in cleaned
    assert "5.2.2 Access and Mobility Management Function (AMF)" in cleaned
    assert "The AMF includes the following functionality: Termination of RAN CP interface (N2)." in cleaned


def test_parse_toc_entry():
    sec_num, sec_title = parse_toc_entry("5.2.2 Access and Mobility Management Function (AMF)")
    assert sec_num == "5.2.2"
    assert sec_title == "Access and Mobility Management Function (AMF)"

    annex_num, annex_title = parse_toc_entry("Annex A (informative): Relationship between interfaces")
    assert "Annex A" in annex_num
    assert "Relationship between interfaces" in annex_title


def test_per_specification_and_merged_jsonl():
    ts501_file = settings.PROCESSED_DATA_DIR / "ts23501_chunks.jsonl"
    ts502_file = settings.PROCESSED_DATA_DIR / "ts23502_chunks.jsonl"
    merged_file = settings.PROCESSED_DATA_DIR / "chunks.jsonl"

    assert ts501_file.exists(), "ts23501_chunks.jsonl should exist"
    assert ts502_file.exists(), "ts23502_chunks.jsonl should exist"
    assert merged_file.exists(), "chunks.jsonl should exist"

    with open(ts501_file, "r", encoding="utf-8") as f:
        chunks_501 = [json.loads(line) for line in f]
    with open(ts502_file, "r", encoding="utf-8") as f:
        chunks_502 = [json.loads(line) for line in f]
    with open(merged_file, "r", encoding="utf-8") as f:
        chunks_merged = [json.loads(line) for line in f]

    assert len(chunks_501) > 1000, f"Expected >1000 chunks in TS 23.501, got {len(chunks_501)}"
    assert len(chunks_502) > 1000, f"Expected >1000 chunks in TS 23.502, got {len(chunks_502)}"
    assert len(chunks_merged) == len(chunks_501) + len(chunks_502)

    # Validate schema of sample chunks
    for raw in chunks_merged[:50]:
        chunk = Chunk.model_validate(raw)
        assert chunk.metadata.document_code in ["3GPP TS 23.501", "3GPP TS 23.502"]
        assert chunk.metadata.start_page > 0
        assert chunk.metadata.end_page >= chunk.metadata.start_page
        assert len(chunk.metadata.page_number) > 0
        assert len(chunk.metadata.section_number) > 0
        assert len(chunk.text.strip()) >= 60
        assert "Document:" in chunk.enriched_text
        assert "Clause:" in chunk.enriched_text
        assert "Page:" in chunk.enriched_text
