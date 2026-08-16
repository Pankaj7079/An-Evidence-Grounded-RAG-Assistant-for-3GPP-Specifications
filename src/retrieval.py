"""All-in-One Native Qdrant Hybrid Retrieval Engine with Server-Side RRF Fusion.

Features:
1. Native Qdrant Multi-Vector Query: Dense (HNSW Cosine) + Sparse (BM25) searched directly in Qdrant.
2. Server-Side Reciprocal Rank Fusion (RRF): Executed directly on Qdrant via Prefetch & FusionQuery.
3. Payload Metadata Extraction: Returns fully populated Chunk payloads directly from Qdrant.
4. Calibrated Evidence Gate: Evaluates retrieval confidence and triggers controlled abstention.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

from config import settings
from src.indexing import get_qdrant_client
from src.models import EvidenceGateDecision, RetrievalResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class HybridRetriever:
    """All-in-One Qdrant Hybrid Retriever using native Dense + Sparse Named Vectors and RRF."""

    def __init__(
        self,
        qdrant_client: Optional[QdrantClient] = None,
        dense_model_name: Optional[str] = None,
    ):
        self.client = qdrant_client or get_qdrant_client()
        self.dense_model_name = dense_model_name or settings.EMBEDDING_MODEL
        self.dense_model = SentenceTransformer(self.dense_model_name)
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def _build_filter(
        self,
        filter_doc: Optional[str] = None,
        filter_type: Optional[str] = None,
    ) -> Optional[rest.Filter]:
        """Construct Qdrant payload filter."""
        conditions = []
        if filter_doc:
            conditions.append(
                rest.FieldCondition(
                    key="document_code",
                    match=rest.MatchValue(value=filter_doc),
                )
            )
        if filter_type:
            conditions.append(
                rest.FieldCondition(
                    key="content_type",
                    match=rest.MatchValue(value=filter_type),
                )
            )
        return rest.Filter(must=conditions) if conditions else None

    def dense_search(
        self,
        query: str,
        top_k: int = 15,
        filter_doc: Optional[str] = None,
        filter_type: Optional[str] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Perform dense vector search in Qdrant; returns list of (chunk_id, cosine_score, payload)."""
        query_vector = self.dense_model.encode(query, normalize_embeddings=True).tolist()
        qdrant_filter = self._build_filter(filter_doc, filter_type)

        response = self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=query_vector,
            using="dense",
            query_filter=qdrant_filter,
            limit=top_k,
        )

        results = []
        for hit in response.points:
            chunk_id = hit.payload.get("chunk_id", str(hit.id))
            results.append((chunk_id, float(hit.score), hit.payload))
        return results

    def sparse_search(
        self,
        query: str,
        top_k: int = 15,
        filter_doc: Optional[str] = None,
        filter_type: Optional[str] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Perform native BM25 sparse search in Qdrant; returns list of (chunk_id, bm25_score, payload)."""
        sparse_vecs = list(self.sparse_model.embed([query]))
        if not sparse_vecs or len(sparse_vecs[0].indices) == 0:
            return []

        s_vec = sparse_vecs[0]
        qdrant_filter = self._build_filter(filter_doc, filter_type)

        response = self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=rest.SparseVector(
                indices=s_vec.indices.tolist(),
                values=s_vec.values.tolist(),
            ),
            using="sparse",
            query_filter=qdrant_filter,
            limit=top_k,
        )

        results = []
        for hit in response.points:
            chunk_id = hit.payload.get("chunk_id", str(hit.id))
            results.append((chunk_id, float(hit.score), hit.payload))
        return results

    def retrieve(
        self,
        query: str,
        top_k: int = settings.CANDIDATE_K,
        filter_doc: Optional[str] = None,
        filter_type: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Perform Native Qdrant Server-Side Hybrid Search with Reciprocal Rank Fusion (RRF)."""
        dense_vector = self.dense_model.encode(query, normalize_embeddings=True).tolist()
        sparse_vecs = list(self.sparse_model.embed([query]))
        qdrant_filter = self._build_filter(filter_doc, filter_type)

        candidate_k = max(top_k * 2, 20)
        prefetch_queries = [
            rest.Prefetch(
                query=dense_vector,
                using="dense",
                limit=candidate_k,
                filter=qdrant_filter,
            )
        ]

        if sparse_vecs and len(sparse_vecs[0].indices) > 0:
            s_vec = sparse_vecs[0]
            prefetch_queries.append(
                rest.Prefetch(
                    query=rest.SparseVector(
                        indices=s_vec.indices.tolist(),
                        values=s_vec.values.tolist(),
                    ),
                    using="sparse",
                    limit=candidate_k,
                    filter=qdrant_filter,
                )
            )

        # Native Qdrant Server-Side RRF Fusion
        response = self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            prefetch=prefetch_queries,
            query=rest.FusionQuery(fusion=rest.Fusion.RRF),
            limit=top_k,
        )

        results: List[RetrievalResult] = []
        for hit in response.points:
            payload = hit.payload
            results.append(
                RetrievalResult(
                    chunk_id=payload.get("chunk_id", str(hit.id)),
                    text=payload.get("text", ""),
                    enriched_text=payload.get("enriched_text", ""),
                    score=round(float(hit.score), 4),
                    document_code=payload.get("document_code", ""),
                    section_number=payload.get("section_number", ""),
                    section_title=payload.get("section_title", ""),
                    section_hierarchy=payload.get("section_hierarchy", ""),
                    start_page=payload.get("start_page", 0),
                    end_page=payload.get("end_page", 0),
                    page_number=str(payload.get("page_number", "")),
                    release=payload.get("release", "17"),
                    version=payload.get("version", "17.4.0"),
                    content_type=payload.get("content_type", "paragraph"),
                    retrieval_method="qdrant_native_rrf",
                )
            )

        return results

    def evaluate_evidence_gate(
        self,
        query: str,
        top_k: int = settings.CANDIDATE_K,
        min_cosine_threshold: float = settings.MIN_RELEVANCE_SCORE,
    ) -> EvidenceGateDecision:
        """Evaluate evidence quality and determine whether grounding is sufficient to answer."""
        dense_hits = self.dense_search(query, top_k=5)
        sparse_hits = self.sparse_search(query, top_k=5)

        top_cosine = dense_hits[0][1] if dense_hits else 0.0
        has_sparse_match = len(sparse_hits) > 0 and sparse_hits[0][1] > 0

        # Calibrated evidence grounding logic:
        # 1. High dense cosine (>= 0.35) -> Pass.
        # 2. Moderate dense cosine (>= 0.28) + sparse telecom keyword match -> Pass.
        # 3. Low confidence / Out-of-domain query -> Fail and trigger controlled abstention.
        is_grounded = (top_cosine >= min_cosine_threshold) or (top_cosine >= 0.28 and has_sparse_match)

        if not is_grounded:
            return EvidenceGateDecision(
                is_sufficient=False,
                reason=(
                    f"Evidence confidence below threshold (dense similarity: {top_cosine:.3f}, "
                    f"sparse keyword match: {has_sparse_match}). Abstaining."
                ),
                top_score=round(top_cosine, 4),
                num_chunks=0,
                retrieved_chunks=[],
            )

        retrieved_chunks = self.retrieve(query, top_k=top_k)
        return EvidenceGateDecision(
            is_sufficient=True,
            reason=f"Sufficient evidence found (dense similarity: {top_cosine:.3f}, retrieved: {len(retrieved_chunks)} chunks).",
            top_score=round(top_cosine, 4),
            num_chunks=len(retrieved_chunks),
            retrieved_chunks=retrieved_chunks,
        )


if __name__ == "__main__":
    retriever = HybridRetriever()
    test_queries = [
        "What are the functions of the AMF in 5G architecture?",
        "Explain the Registration Procedure step by step in TS 23.502",
        "What is the capital of France?",
    ]

    for q in test_queries:
        print("\n" + "=" * 70)
        print(f"QUERY: '{q}'")
        print("=" * 70)
        decision = retriever.evaluate_evidence_gate(q)
        print(f"Evidence Gate: {'PASS (Sufficient)' if decision.is_sufficient else 'FAIL (Abstain)'}")
        print(f"Reason:        {decision.reason}")
        print(f"Top Score:     {decision.top_score}")
