"""All-in-One Native Qdrant Hybrid Indexing Engine for 3GPP Specifications.

Features:
1. Native Qdrant Multi-Vector Collection:
   - 'dense': 384-dim HNSW Cosine vector (sentence-transformers/all-MiniLM-L6-v2)
   - 'sparse': Native BM25 sparse vector (fastembed Qdrant/bm25)
2. Payload Metadata: Full metadata (document_code, section_number, page_number, text, etc.)
3. Dual-Mode Storage: Seamlessly connects to Qdrant Cloud or local embedded Qdrant.
4. Zero local index files: Everything lives 100% inside Qdrant!
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

from config import settings
from src.models import Chunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    """Return QdrantClient configured for Qdrant Cloud or local disk fallback."""
    if settings.QDRANT_URL and settings.QDRANT_API_KEY:
        logger.info(f"Connecting to Qdrant Cloud: {settings.QDRANT_URL}")
        return QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=60.0,
        )
    else:
        logger.info(f"Using local embedded Qdrant storage: {settings.QDRANT_PATH}")
        settings.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(settings.QDRANT_PATH))


def setup_qdrant_hybrid_collection(client: QdrantClient, vector_dim: int = 384) -> None:
    """Create or recreate Qdrant collection with named dense + sparse vectors and HNSW graph."""
    collection_name = settings.COLLECTION_NAME

    if client.collection_exists(collection_name):
        logger.info(f"Recreating hybrid collection '{collection_name}'...")
        client.delete_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": rest.VectorParams(
                size=vector_dim,
                distance=rest.Distance.COSINE,
                hnsw_config=rest.HnswConfigDiff(
                    m=16,
                    ef_construct=128,
                    full_scan_threshold=1000,
                ),
            )
        },
        sparse_vectors_config={
            "sparse": rest.SparseVectorParams(
                index=rest.SparseIndexParams(
                    on_disk=False,
                )
            )
        },
    )

    # Set up payload field indexes for instant metadata filtering
    payload_indexes = [
        ("document_code", rest.PayloadSchemaType.KEYWORD),
        ("section_number", rest.PayloadSchemaType.KEYWORD),
        ("content_type", rest.PayloadSchemaType.KEYWORD),
        ("page_number", rest.PayloadSchemaType.KEYWORD),
        ("start_page", rest.PayloadSchemaType.INTEGER),
        ("end_page", rest.PayloadSchemaType.INTEGER),
    ]

    for field_name, field_schema in payload_indexes:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception:
            pass  # Local embedded Qdrant automatically indexes payloads

    logger.info(f"Qdrant collection '{collection_name}' initialized with Dense (HNSW) + Sparse (BM25) vectors.")


def index_chunks(
    chunks_path: Optional[Path] = None,
    batch_size: int = 64,
) -> int:
    """Load chunks, compute dense + sparse embeddings, and index directly into Qdrant."""
    if chunks_path is None:
        chunks_path = settings.PROCESSED_DATA_DIR / "chunks.jsonl"

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found at {chunks_path}. Run ingestion first.")

    logger.info(f"Loading chunks from {chunks_path}...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = [Chunk.model_validate(json.loads(line)) for line in f]

    total_chunks = len(chunks)
    logger.info(f"Loaded {total_chunks} chunks.")

    # 1. Initialize Encoders
    logger.info(f"Loading Dense embedding model: {settings.EMBEDDING_MODEL}...")
    dense_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    vector_dim = dense_model.get_sentence_embedding_dimension() if hasattr(dense_model, "get_sentence_embedding_dimension") else 384

    logger.info("Loading Native Sparse BM25 embedding model (Qdrant/bm25)...")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    # 2. Setup Qdrant Hybrid Collection
    client = get_qdrant_client()
    setup_qdrant_hybrid_collection(client, vector_dim=vector_dim)

    # 3. Compute Embeddings & Upload in Batches
    logger.info(f"Computing Dense + Sparse embeddings and uploading to Qdrant in batches of {batch_size}...")
    texts_to_embed = [c.enriched_text for c in chunks]

    for i in range(0, total_chunks, batch_size):
        batch_end = min(i + batch_size, total_chunks)
        batch_chunks = chunks[i:batch_end]
        batch_texts = texts_to_embed[i:batch_end]

        # Compute Dense vectors
        dense_embeddings = dense_model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )

        # Compute Sparse BM25 vectors
        sparse_embeddings = list(sparse_model.embed(batch_texts))

        points = []
        for idx, (chunk, d_vec, s_vec) in enumerate(zip(batch_chunks, dense_embeddings, sparse_embeddings)):
            point_id = i + idx
            payload = {
                "chunk_id": chunk.metadata.chunk_id,
                "document_code": chunk.metadata.document_code,
                "document_title": chunk.metadata.document_title,
                "release": chunk.metadata.release,
                "version": chunk.metadata.version,
                "section_number": chunk.metadata.section_number,
                "section_title": chunk.metadata.section_title,
                "section_hierarchy": chunk.metadata.section_hierarchy,
                "start_page": chunk.metadata.start_page,
                "end_page": chunk.metadata.end_page,
                "page_number": chunk.metadata.page_number,
                "content_type": chunk.metadata.content_type,
                "text": chunk.text,
                "enriched_text": chunk.enriched_text,
            }
            points.append(
                rest.PointStruct(
                    id=point_id,
                    vector={
                        "dense": d_vec.tolist(),
                        "sparse": rest.SparseVector(
                            indices=s_vec.indices.tolist(),
                            values=s_vec.values.tolist(),
                        ),
                    },
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=points,
        )

        if (i + batch_size) % 512 < batch_size or batch_end == total_chunks:
            logger.info(f"Uploaded {batch_end}/{total_chunks} chunks to Qdrant ({batch_end/total_chunks*100:.1f}%)")

    logger.info(f"Successfully indexed all {total_chunks} chunks with Dense + Sparse vectors in Qdrant!")
    return total_chunks


if __name__ == "__main__":
    index_chunks()
