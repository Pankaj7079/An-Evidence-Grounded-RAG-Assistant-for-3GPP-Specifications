"""End-to-End Evidence-Grounded RAG Pipeline with Two-Stage Retrieval & Re-ranking.

Pipeline Flow:
1. Evidence Gate Evaluation (Pre-retrieval confidence check)
2. Stage 1: Native Qdrant Multi-Vector Hybrid Search (Dense + Sparse BM25 -> Top 15 Candidates)
3. Stage 2: Cross-Encoder Re-ranking (ms-marco-MiniLM-L-6-v2 -> Top 4 High-Precision Chunks)
4. Grounded Prompt Formulation (Strict context boundaries)
5. Deterministic LLM Generation (Groq Llama-3.3-70b / Gemini @ temp=0.0)
6. Post-Generation Citation Validation (Zero-hallucination verification)
"""

import logging
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from config import settings
from src.citation_validator import CitationValidationResult, CitationValidator
from src.generation import LLMClient, format_grounded_prompt
from src.models import EvidenceGateDecision, RetrievalResult
from src.reranker import CrossEncoderReranker
from src.retrieval import HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PipelineResponse(BaseModel):
    """Complete structured response from the 3GPP Spec Assistant pipeline."""

    query: str
    answer: str
    is_abstained: bool = False
    evidence_gate: EvidenceGateDecision
    citation_validation: Optional[CitationValidationResult] = None
    retrieved_chunks: List[RetrievalResult] = Field(default_factory=list)
    candidate_count: int = 0
    latency_ms: float = 0.0
    llm_provider: str = settings.LLM_PROVIDER


class RAGPipeline:
    """Production Two-Stage Evidence-Grounded 3GPP Spec Assistant RAG Pipeline."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.llm_client = llm_client or LLMClient()
        self.validator = CitationValidator()

    def query(
        self,
        question: str,
        filter_doc: Optional[str] = None,
        filter_type: Optional[str] = None,
        candidate_k: int = settings.CANDIDATE_K,
        final_k: int = settings.FINAL_CONTEXT_K,
    ) -> PipelineResponse:
        """Process user question through Two-Stage Retrieval, Re-ranking, and Generation."""
        start_time = time.perf_counter()
        clean_question = question.strip()

        # Step 1: Evidence Gate Check
        gate_decision = self.retriever.evaluate_evidence_gate(clean_question, top_k=candidate_k)

        if not gate_decision.is_sufficient:
            latency = (time.perf_counter() - start_time) * 1000
            logger.info(f"Evidence Gate rejected query: '{clean_question}' (Score: {gate_decision.top_score})")
            return PipelineResponse(
                query=clean_question,
                answer=settings.ABSTENTION_MESSAGE,
                is_abstained=True,
                evidence_gate=gate_decision,
                citation_validation=None,
                retrieved_chunks=[],
                candidate_count=0,
                latency_ms=round(latency, 2),
                llm_provider=self.llm_client.provider,
            )

        # Step 2: Stage 1 - Native Qdrant Multi-Vector Hybrid Retrieval (Dense + Sparse BM25)
        candidates = self.retriever.retrieve(
            query=clean_question,
            top_k=candidate_k,
            filter_doc=filter_doc,
            filter_type=filter_type,
        )

        # Step 3: Stage 2 - Cross-Encoder Re-ranking
        top_chunks = self.reranker.rerank(
            query=clean_question,
            candidates=candidates,
            top_k=final_k,
        )

        # Step 4: Grounded Prompt Formulation
        user_prompt = format_grounded_prompt(
            query=clean_question,
            retrieved_chunks=top_chunks,
        )

        # Step 5: Deterministic LLM Generation
        raw_answer = self.llm_client.generate(user_prompt)
        is_llm_abstained = (
            raw_answer.strip().lower().startswith("i could not find sufficient supporting evidence")
            or (len(raw_answer.strip()) < 150 and "could not find sufficient supporting evidence" in raw_answer.lower())
        )

        # Step 6: Post-Generation Citation Validation
        validation_result = self.validator.validate(raw_answer, top_chunks)

        latency = (time.perf_counter() - start_time) * 1000

        return PipelineResponse(
            query=clean_question,
            answer=raw_answer,
            is_abstained=is_llm_abstained,
            evidence_gate=gate_decision,
            citation_validation=validation_result,
            retrieved_chunks=top_chunks,
            candidate_count=len(candidates),
            latency_ms=round(latency, 2),
            llm_provider=self.llm_client.provider,
        )


if __name__ == "__main__":
    pipeline = RAGPipeline()
    test_queries = [
        "What are the main functions of the AMF in 5G system architecture?",
        "Explain the Registration Procedure in 3GPP TS 23.502",
        "What is the capital of France?",
    ]

    for q in test_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: '{q}'")
        print("=" * 80)
        res = pipeline.query(q)
        print(f"Abstained:          {res.is_abstained}")
        print(f"Latency:            {res.latency_ms:.1f} ms")
        print(f"Gate Decision:      {res.evidence_gate.reason}")
        print(f"Candidates Filtered:{res.candidate_count} -> {len(res.retrieved_chunks)} final context chunks")
        if res.citation_validation:
            print(f"Citation Precision: {res.citation_validation.citation_precision * 100:.1f}% ({len(res.citation_validation.valid_citations)} valid / {res.citation_validation.total_citations} total)")
        print("\n[GENERATED ANSWER]:")
        print(res.answer)
