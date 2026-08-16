"""Unit tests for citation validation and prompt formatting."""

import pytest
from src.citation_validator import CitationValidator
from src.generation import format_grounded_prompt
from src.models import RetrievalResult


def test_citation_extraction():
    sample_text = (
        "The AMF terminates N2 interface [TS 23.501 Clause 5.2.2, Page 42]. "
        "The SMF controls UPF via N4 [TS 23.501 Clause 4.2.4, Page 42-43]. "
        "Registration procedure is defined in [3GPP TS 23.502 Clause 4.2.2.2, Page 25]."
    )
    citations = CitationValidator.extract_citations(sample_text)
    assert len(citations) == 3

    assert citations[0].document_code == "TS 23.501"
    assert citations[0].section_number == "5.2.2"
    assert citations[0].page_number == "42"

    assert citations[1].document_code == "TS 23.501"
    assert citations[1].section_number == "4.2.4"
    assert citations[1].page_number == "42-43"

    assert citations[2].document_code == "TS 23.502"
    assert citations[2].section_number == "4.2.2.2"
    assert citations[2].page_number == "25"


def test_citation_validation_against_retrieved_context():
    mock_retrieved = [
        RetrievalResult(
            chunk_id="chunk_1",
            text="AMF functionality",
            enriched_text="",
            score=0.9,
            document_code="3GPP TS 23.501",
            section_number="6.2.1",
            section_title="AMF",
            page_number="423-424",
            release="17",
            version="17.4.0",
        ),
        RetrievalResult(
            chunk_id="chunk_2",
            text="Registration flow",
            enriched_text="",
            score=0.85,
            document_code="3GPP TS 23.502",
            section_number="4.2.2.2",
            section_title="Registration",
            page_number="25",
            release="16",
            version="16.9.0",
        ),
    ]

    valid_text = "The AMF manages mobility [TS 23.501 Clause 6.2.1, Page 423-424]."
    result_valid = CitationValidator.validate(valid_text, mock_retrieved)
    assert result_valid.is_valid is True
    assert result_valid.citation_precision == 1.0
    assert len(result_valid.valid_citations) == 1
    assert len(result_valid.invalid_citations) == 0

    hallucinated_text = "The AMF manages 6G satellites [TS 23.501 Clause 99.99, Page 999]."
    result_invalid = CitationValidator.validate(hallucinated_text, mock_retrieved)
    assert result_invalid.is_valid is False
    assert result_invalid.citation_precision == 0.0
    assert len(result_invalid.invalid_citations) == 1


def test_prompt_formatting():
    mock_chunks = [
        RetrievalResult(
            chunk_id="c1",
            text="Test text",
            enriched_text="",
            score=0.8,
            document_code="3GPP TS 23.501",
            section_number="4.2.4",
            section_title="Roaming",
            page_number="42-43",
            release="17",
            version="17.4.0",
        )
    ]
    prompt = format_grounded_prompt("What is roaming?", mock_chunks)
    assert "3GPP TS 23.501" in prompt
    assert "Clause: 4.2.4" in prompt
    assert "Page Number: 42-43" in prompt
    assert "What is roaming?" in prompt
