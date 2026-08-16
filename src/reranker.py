"""Cross-Encoder re-ranker for precise 3GPP clause ranking."""

import logging
from typing import List, Optional
from sentence_transformers import CrossEncoder

from config import settings
from src.models import RetrievalResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Stage 2 re-ranker using cross-attention transformer scoring."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        logger.info(f"Loading Cross-Encoder model: {self.model_name}")
        # Initialize cross-encoder model
        self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = settings.FINAL_CONTEXT_K,
    ) -> List[RetrievalResult]:
        """Score candidate chunks with cross-attention and return top_k results."""
        if not candidates:
            return []

        # Return early if candidates are already within top_k
        if len(candidates) <= top_k:
            return candidates

        # Build (query, text) pairs with document and clause header context
        pairs = [
            (
                query,
                f"{c.document_code} Clause {c.section_number} ({c.section_title})\n{c.text}",
            )
            for c in candidates
        ]

        # Compute cross-encoder similarity scores
        scores = self.model.predict(pairs)

        # Attach computed re-ranking scores to candidate chunks
        scored_candidates = []
        for chunk, score in zip(candidates, scores):
            chunk_copy = chunk.model_copy()
            chunk_copy.rerank_score = round(float(score), 4)
            scored_candidates.append(chunk_copy)

        # Sort candidates descending by cross-encoder score
        scored_candidates.sort(key=lambda x: (x.rerank_score or -999.0), reverse=True)
        top_results = scored_candidates[:top_k]

        logger.debug(
            f"Re-ranked {len(candidates)} candidates down to {len(top_results)} (Top score: {top_results[0].rerank_score})"
        )
        return top_results
