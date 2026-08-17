"""End-to-end evidence-grounded RAG pipeline orchestration."""

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
    """Structured response from the RAG pipeline."""

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
    """Production Two-Stage Evidence-Grounded RAG Pipeline for 3GPP Specifications."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        # Initialize pipeline components
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.llm_client = llm_client or LLMClient()
        self.validator = CitationValidator()

    def warmup(self) -> None:
        """Pre-warm PyTorch neural models, tokenizers, and cloud connections."""
        try:
            # 1. Warm up dense embedding model
            _ = self.retriever.dense_model.encode("3gpp 5g", normalize_embeddings=True)
            # 2. Warm up sparse BM25 tokenizer
            _ = list(self.retriever.sparse_model.embed(["3gpp 5g"]))
            # 3. Warm up Cross-Encoder transformer
            _ = self.reranker.model.predict([("3gpp 5g", "3gpp 5g architecture")])
            # 4. Pre-warm Qdrant Cloud connection pool
            _ = self.retriever.client.collection_exists(settings.COLLECTION_NAME)
            logger.info("RAG pipeline neural models and connections pre-warmed successfully.")
        except Exception as e:
            logger.debug(f"Warmup notice: {e}")

    def query(
        self,
        question: str,
        filter_doc: Optional[str] = None,
        filter_type: Optional[str] = None,
        candidate_k: int = settings.CANDIDATE_K,
        final_k: int = settings.FINAL_CONTEXT_K,
    ) -> PipelineResponse:
        """Process user question through Retrieval, Re-ranking, Generation, and Validation."""
        start_time = time.perf_counter()
        clean_question = question.strip()

        # Step 1: Hybrid retrieval and evidence gate check
        gate_decision, candidate_chunks = self.retriever.retrieve_with_gate(
            query=clean_question,
            top_k=candidate_k,
            filter_doc=filter_doc,
            filter_type=filter_type,
            min_cosine_threshold=settings.MIN_RELEVANCE_SCORE,
        )

        # Step 2: Immediate abstention if evidence gate fails
        if not gate_decision.is_sufficient or not candidate_chunks:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            abstain_text = (
                "I could not find sufficient supporting evidence in the indexed 3GPP documents "
                f"for this query."
            )
            return PipelineResponse(
                query=clean_question,
                answer=abstain_text,
                retrieved_chunks=[],
                candidate_count=0,
                evidence_gate=gate_decision,
                citation_validation=None,
                latency_ms=round(elapsed_ms, 2),
                llm_provider=self.llm_client.provider,
                is_abstained=True,
            )

        # Step 3: Cross-Encoder transformer re-ranking
        top_chunks = self.reranker.rerank(
            query=clean_question,
            candidates=candidate_chunks,
            top_k=final_k,
        )

        # Step 4: Format grounded prompt with clause excerpts
        user_prompt = format_grounded_prompt(
            query=clean_question,
            retrieved_chunks=top_chunks,
        )

        # Step 5: Deterministic LLM response generation
        raw_answer = self.llm_client.generate(user_prompt)
        is_llm_abstained = (
            raw_answer.strip().lower().startswith("i could not find sufficient supporting evidence")
            or (len(raw_answer.strip()) < 150 and "could not find sufficient supporting evidence" in raw_answer.lower())
        )

        # Step 6: Validate citations against retrieved sources
        validation_result = self.validator.validate(raw_answer, top_chunks)

        latency = (time.perf_counter() - start_time) * 1000

        return PipelineResponse(
            query=clean_question,
            answer=raw_answer,
            is_abstained=is_llm_abstained,
            evidence_gate=gate_decision,
            citation_validation=validation_result,
            retrieved_chunks=top_chunks,
            candidate_count=len(candidate_chunks),
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
