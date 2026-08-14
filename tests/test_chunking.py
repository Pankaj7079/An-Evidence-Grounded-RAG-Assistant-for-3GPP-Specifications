"""Unit tests for PDF ingestion and clause-aware chunking."""

import json
from pathlib import Path
import pytest
from config import settings
from src.ingestion import clean_page_text, is_table_of_contents_page
from src.models import Chunk, ChunkMetadata


def test_clean_page_text():
    sample_text = (
        "ETSI\n"
        "ETSI TS 123 501 V17.4.0 (2022-05)\n"
        "42\n"
        "3GPP TS 23.501 version 17.4.0 Release 17\n"
        "5.2.2 Access and Mobility Management Function (AMF)\n"
        "The AMF includes the following functionality: Termination of RAN CP interface (N2)."
    )
    cleaned = clean_page_text(sample_text)
    assert "ETSI TS 123 501" not in cleaned
    assert "version 17.4.0 Release 17" not in cleaned
    assert "5.2.2 Access and Mobility Management Function (AMF)" in cleaned
    assert "Termination of RAN CP interface (N2)" in cleaned


def test_is_table_of_contents_page():
    toc_text = (
        "Contents\n"
        "Foreword ................................................................ 19\n"
        "1 Scope ................................................................ 20\n"
        "2 References ........................................................... 20\n"
    )
    assert is_table_of_contents_page(toc_text) is True

    body_text = (
        "5.2.2 Access and Mobility Management Function (AMF)\n"
        "The AMF includes the following functionality:\n"
        "- Termination of RAN CP interface (N2).\n"
        "- Termination of NAS (N1), NAS ciphering and integrity protection."
    )
    assert is_table_of_contents_page(body_text) is False


def test_processed_chunks_validity():
    chunks_file = settings.PROCESSED_DATA_DIR / "chunks.jsonl"
    assert chunks_file.exists(), "Processed chunks JSONL file should exist"

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = [json.loads(line) for line in f]

    assert len(chunks_data) > 1000, f"Expected >1000 chunks, found {len(chunks_data)}"

    # Test individual chunk validation with Pydantic
    for raw in chunks_data[:50]:
        chunk = Chunk.model_validate(raw)
        assert chunk.metadata.document_code in ["3GPP TS 23.501", "3GPP TS 23.502"]
        assert chunk.metadata.page_number > 0
        assert len(chunk.metadata.section_number) > 0
        assert len(chunk.text.strip()) > 0
        assert "Document:" in chunk.enriched_text
        assert "Clause:" in chunk.enriched_text
        assert "Page:" in chunk.enriched_text
