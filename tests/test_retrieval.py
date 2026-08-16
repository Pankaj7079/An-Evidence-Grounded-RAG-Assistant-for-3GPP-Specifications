"""Unit and integration tests for native Qdrant hybrid retrieval, RRF ranking, re-ranker, and evidence gate."""

import pytest
from config import settings
from src.reranker import CrossEncoderReranker
from src.retrieval import HybridRetriever


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever()


@pytest.fixture(scope="module")
def reranker():
    return CrossEncoderReranker()


def test_native_sparse_search(retriever):
    hits = retriever.sparse_search("AMF termination of N2 interface", top_k=5)
    assert len(hits) > 0
    chunk_id, score, payload = hits[0]
    assert score > 0
    assert "document_code" in payload
    assert "section_number" in payload


def test_native_dense_vector_search(retriever):
    hits = retriever.dense_search("5G System Architecture Model", top_k=5)
    assert len(hits) > 0
    chunk_id, score, payload = hits[0]
    assert score > 0.30
    assert "document_code" in payload


def test_native_hybrid_retrieval_rrf(retriever):
    results = retriever.retrieve("Functions of the Access and Mobility Management Function AMF", top_k=5)
    assert len(results) == 5
    top = results[0]
    assert top.score > 0.0
    assert top.document_code in ["3GPP TS 23.501", "3GPP TS 23.502"]
    assert top.section_number != ""
    assert top.page_number != ""
    assert len(top.text) > 0


def test_cross_encoder_reranking(retriever, reranker):
    query = "What are the core functions of the AMF in 5G architecture?"
    candidates = retriever.retrieve(query, top_k=10)
    assert len(candidates) == 10

    reranked = reranker.rerank(query, candidates, top_k=4)
    assert len(reranked) == 4
    for r in reranked:
        assert r.rerank_score is not None
    # Verify scores are sorted descending
    for i in range(len(reranked) - 1):
        assert reranked[i].rerank_score >= reranked[i + 1].rerank_score


def test_retrieval_metadata_filtering(retriever):
    results_501 = retriever.retrieve("Architecture concepts", top_k=5, filter_doc="3GPP TS 23.501")
    for r in results_501:
        assert r.document_code == "3GPP TS 23.501"

    results_502 = retriever.retrieve("Registration procedure", top_k=5, filter_doc="3GPP TS 23.502")
    for r in results_502:
        assert r.document_code == "3GPP TS 23.502"


def test_evidence_gate_in_domain(retriever):
    decision = retriever.evaluate_evidence_gate("What are the functions of the AMF in 5G architecture?")
    assert decision.is_sufficient is True
    assert decision.top_score >= 0.35
    assert len(decision.retrieved_chunks) > 0


def test_evidence_gate_out_of_domain_abstention(retriever):
    decision = retriever.evaluate_evidence_gate("What is the capital of France?")
    assert decision.is_sufficient is False
    assert decision.top_score < 0.30
    assert len(decision.retrieved_chunks) == 0
