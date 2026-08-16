"""Data models for chunks, metadata, citations, and retrieval results."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata for a single 3GPP document chunk."""

    chunk_id: str = Field(..., description="Unique deterministic identifier for the chunk")
    document_code: str = Field(..., description="Specification code, e.g. 3GPP TS 23.501")
    document_title: str = Field(..., description="Full title of the specification")
    release: str = Field(..., description="3GPP Release, e.g. 17 or 16")
    version: str = Field(..., description="Specification version, e.g. 17.4.0")
    section_number: str = Field(..., description="Clause number, e.g. 6.2.1")
    section_title: str = Field(..., description="Clause title, e.g. AMF")
    section_hierarchy: str = Field(
        default="",
        description="Full clause hierarchy breadcrumbs",
    )
    start_page: int = Field(..., description="First printed page of the chunk")
    end_page: int = Field(..., description="Last printed page of the chunk")
    page_number: str = Field(
        ...,
        description="Printed page string or range (e.g. '423' or '423-424')",
    )
    content_type: Literal["paragraph", "table", "procedure_step", "figure_note", "annex"] = Field(
        default="paragraph",
        description="Type of content contained in this chunk",
    )


class Chunk(BaseModel):
    """Structured text chunk extracted from 3GPP specifications."""

    metadata: ChunkMetadata
    text: str = Field(..., description="Raw text extracted from PDF")
    enriched_text: str = Field(
        ...,
        description="Text prepended with document, clause, and hierarchy metadata for dense search",
    )

    def to_enriched_string(self) -> str:
        """Format the chunk with contextual header metadata for embedding and search."""
        hierarchy_line = f"Hierarchy: {self.metadata.section_hierarchy}\n" if self.metadata.section_hierarchy else ""
        type_line = f"Content Type: {self.metadata.content_type}\n" if self.metadata.content_type != "paragraph" else ""

        return (
            f"Document: {self.metadata.document_code}\n"
            f"Title: {self.metadata.document_title}\n"
            f"Release: {self.metadata.release}\n"
            f"{hierarchy_line}"
            f"Clause: {self.metadata.section_number}\n"
            f"Clause title: {self.metadata.section_title}\n"
            f"Page: {self.metadata.page_number}\n"
            f"{type_line}"
            f"Content:\n{self.text.strip()}"
        )


class Citation(BaseModel):
    """Citation referencing a 3GPP specification clause and page."""

    document_code: str = Field(..., description="Specification code, e.g. TS 23.501")
    section_number: str = Field(..., description="Clause number, e.g. 6.2.1")
    page_number: str = Field(..., description="Page number or range, e.g. 423-424")
    raw_citation: str = Field(..., description="Raw citation string parsed from response")


class RetrievalResult(BaseModel):
    """Retrieval result returned from hybrid search and re-ranking."""

    chunk_id: str
    text: str
    enriched_text: str
    score: float
    document_code: str
    section_number: str
    section_title: str
    section_hierarchy: str = ""
    start_page: int = 0
    end_page: int = 0
    page_number: str = ""
    release: str = "17"
    version: str = "17.4.0"
    content_type: str = "paragraph"
    retrieval_method: str = "qdrant_native_rrf"
    rerank_score: Optional[float] = None


class EvidenceGateDecision(BaseModel):
    """Result of the evidence gating relevance check."""

    is_sufficient: bool
    reason: str
    top_score: float
    confidence_percent: float = Field(default=0.0, description="Confidence percentage (0-100%)")
    num_chunks: int
    retrieved_chunks: List[RetrievalResult] = Field(default_factory=list)
