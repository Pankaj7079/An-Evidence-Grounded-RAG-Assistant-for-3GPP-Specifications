"""Integration tests for end-to-end RAG pipeline."""

import pytest
from src.pipeline import RAGPipeline


@pytest.fixture(scope="module")
def pipeline():
    return RAGPipeline()


def test_pipeline_in_domain_query(pipeline):
    response = pipeline.query("What are the functions of the AMF in 5G architecture?")
    assert response.is_abstained is False
    assert response.evidence_gate.is_sufficient is True
    assert len(response.retrieved_chunks) > 0
    assert len(response.answer) > 50
    assert response.citation_validation is not None
    assert response.citation_validation.citation_precision >= 0.80
    assert response.latency_ms > 0


def test_pipeline_out_of_domain_abstention(pipeline):
    response = pipeline.query("What is the capital of France?")
    assert response.is_abstained is True
    assert response.evidence_gate.is_sufficient is False
    assert "could not find sufficient supporting evidence" in response.answer
    assert len(response.retrieved_chunks) == 0


def test_pipeline_document_filtering(pipeline):
    response = pipeline.query("Registration procedure", filter_doc="3GPP TS 23.502")
    assert response.is_abstained is False
    for chunk in response.retrieved_chunks:
        assert chunk.document_code == "3GPP TS 23.502"
