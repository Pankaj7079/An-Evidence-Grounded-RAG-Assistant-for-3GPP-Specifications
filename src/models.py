"""Pydantic data models for chunks, metadata, relationships, and retrieval."""

from typing import Literal, Optional, List
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata associated with a single document chunk."""

    chunk_id: str = Field(..., description="Unique deterministic identifier for the chunk")
    document_code: str = Field(..., description="e.g. 3GPP TS 23.501")
    document_title: str = Field(..., description="Full title of the specification")
    release: str = Field(..., description="3GPP Release, e.g. 17 or 16")
    version: str = Field(..., description="Exact specification version, e.g. 17.4.0")
    section_number: str = Field(..., description="Clause number, e.g. 5.2.2")
    section_title: str = Field(..., description="Clause title, e.g. Access and Mobility Management Function")
    page_number: int = Field(..., description="Physical 1-indexed page number in the PDF")
    content_type: Literal["paragraph", "table", "procedure_step", "figure_note", "annex"] = Field(
        default="paragraph",
        description="Type of content contained in this chunk",
    )


class Chunk(BaseModel):
    """Structured text chunk extracted from a 3GPP specification."""

    metadata: ChunkMetadata
    text: str = Field(..., description="Original raw extracted clause text")
    enriched_text: str = Field(
        ...,
        description="Text prepended with document, clause, title, and page metadata for embedding & retrieval",
    )

    def to_enriched_string(self) -> str:
        """Return the standardized enriched representation for retrieval."""
        return (
            f"Document: {self.metadata.document_code}\n"
            f"Title: {self.metadata.document_title}\n"
            f"Release: {self.metadata.release}\n"
            f"Clause: {self.metadata.section_number}\n"
            f"Clause title: {self.metadata.section_title}\n"
            f"Page: {self.metadata.page_number}\n"
            f"Content:\n{self.text.strip()}"
        )


class Relationship(BaseModel):
    """Structured network function or procedure relationship."""

    subject: str = Field(..., description="Entity or Network Function (e.g. AMF, SMF)")
    relation: str = Field(
        ...,
        description="Standard relation type (e.g. DEFINED_IN, PARTICIPATES_IN, COMMUNICATES_WITH, CONNECTS_TO, CONTROLS)",
    )
    object: str = Field(..., description="Target entity, interface, or procedure (e.g. UPF, N2, Registration Procedure)")
    document_code: str = Field(..., description="Source document code (e.g. TS 23.501)")
    section_number: str = Field(..., description="Clause where relation is described")
    page_number: int = Field(..., description="PDF page number")
    evidence_text: str = Field(..., description="Exact supporting sentence or excerpt")
    status: Literal["source_verified", "manually_verified"] = "source_verified"


class Citation(BaseModel):
    """Structured citation referencing a 3GPP specification clause and page."""

    document_code: str = Field(..., description="e.g. TS 23.501 or 3GPP TS 23.501")
    section_number: str = Field(..., description="Clause number, e.g. 5.2.2")
    page_number: int = Field(..., description="Page number")
    raw_citation: str = Field(..., description="Exact string parsed from LLM response")


class RetrievalResult(BaseModel):
    """Result returned from hybrid dense + sparse retrieval."""

    chunk_id: str
    text: str
    enriched_text: str
    score: float
    document_code: str
    section_number: str
    section_title: str
    page_number: int
    release: str
    version: str
    retrieval_method: str = "dense_bm25_rrf"


class EvidenceGateDecision(BaseModel):
    """Decision from the evidence gate evaluator."""

    is_sufficient: bool
    reason: str
    top_score: float
    num_chunks: int
    retrieved_chunks: List[RetrievalResult] = Field(default_factory=list)
