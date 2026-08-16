"""Cross-Encoder Re-ranker for High-Precision 3GPP Clause Ranking.

Passes (Query, Candidate Chunk) pairs through a full cross-attention transformer
(cross-encoder/ms-marco-MiniLM-L-6-v2) to accurately rank and filter candidate chunks
from Stage 1 hybrid retrieval into the final top-k context.
"""

import logging
from typing import List, Optional
from sentence_transformers import CrossEncoder

from config import settings
from src.models import RetrievalResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Two-Stage Re-ranker using deep cross-attention transformer scoring."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        logger.info(f"Loading Cross-Encoder Re-ranker: {self.model_name}...")
        self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = settings.FINAL_CONTEXT_K,
    ) -> List[RetrievalResult]:
        """Score candidate chunks with cross-attention and return top_k most relevant chunks."""
        if not candidates:
            return []

        if len(candidates) <= top_k:
            return candidates

        # Form (Query, Text) pairs with specification and clause headers for sharp attention
        pairs = [
            (
                query,
                f"{c.document_code} Clause {c.section_number} ({c.section_title})\n{c.text}",
            )
            for c in candidates
        ]

        # Compute cross-encoder relevance logits
        scores = self.model.predict(pairs)

        # Attach scores and rank
        scored_candidates = []
        for chunk, score in zip(candidates, scores):
            # Create a copy with rerank_score
            chunk_copy = chunk.model_copy()
            chunk_copy.rerank_score = round(float(score), 4)
            scored_candidates.append(chunk_copy)

        scored_candidates.sort(key=lambda x: (x.rerank_score or -999.0), reverse=True)
        top_results = scored_candidates[:top_k]

        logger.debug(
            f"Re-ranked {len(candidates)} candidates down to {len(top_results)} "
            f"(Top score: {top_results[0].rerank_score})"
        )
        return top_results


if __name__ == "__main__":
    from src.retrieval import HybridRetriever

    retriever = HybridRetriever()
    reranker = CrossEncoderReranker()

    test_query = "What are the core functions of the AMF in 5G architecture?"
    print(f"\nQUERY: '{test_query}'")

    # Stage 1: Hybrid Retrieval (15 candidates)
    stage1_candidates = retriever.retrieve(test_query, top_k=15)
    print(f"\nStage 1 retrieved {len(stage1_candidates)} candidates.")

    # Stage 2: Cross-Encoder Re-ranking (Top 4)
    top_chunks = reranker.rerank(test_query, stage1_candidates, top_k=4)
    print(f"\nStage 2 Re-ranked Top {len(top_chunks)} Chunks:")
    for idx, c in enumerate(top_chunks, 1):
        print(f"  #{idx} [Score: {c.rerank_score}] [{c.document_code} Clause {c.section_number}, Page {c.page_number}] {c.section_title}")
