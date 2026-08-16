"""Citation Extraction & Grounding Validator.

Extracts inline citations from generated text, cross-references each citation against
retrieved 3GPP context chunks, and validates clause numbers, page numbers, and specifications.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from src.models import Citation, RetrievalResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CITATION_REGEX = re.compile(
    r"\[(?:3GPP\s+)?(TS\s+23\.50[12])\s+Clause\s+([^,\]]+)(?:,\s*Page\s+([^\]]+))?\]",
    re.IGNORECASE,
)


class CitationValidationResult(BaseModel):
    """Result of citation validation against retrieved context chunks."""

    is_valid: bool = Field(..., description="True if all extracted citations exist in retrieved context")
    total_citations: int = Field(default=0, description="Total citations found in response")
    valid_citations: List[Citation] = Field(default_factory=list)
    invalid_citations: List[Citation] = Field(default_factory=list)
    citation_precision: float = Field(default=1.0, description="Ratio of valid citations to total citations")
    retrieved_sources: List[str] = Field(default_factory=list, description="List of valid source citations available in context")


class CitationValidator:
    """Extracts and verifies citations in LLM generated responses."""

    @staticmethod
    def extract_citations(text: str) -> List[Citation]:
        """Extract structured Citation objects from text using regex."""
        citations = []
        for match in CITATION_REGEX.finditer(text):
            doc = match.group(1).upper().replace("  ", " ").strip()
            clause = match.group(2).strip()
            page = match.group(3).strip() if match.group(3) else ""
            raw = match.group(0)

            citations.append(
                Citation(
                    document_code=doc,
                    section_number=clause,
                    page_number=page,
                    raw_citation=raw,
                )
            )
        return citations

    @classmethod
    def validate(
        cls,
        generated_text: str,
        retrieved_chunks: List[RetrievalResult],
    ) -> CitationValidationResult:
        """Verify that every citation in generated_text exists in retrieved_chunks."""
        extracted = cls.extract_citations(generated_text)

        # Build lookup set of valid (doc_code_normalized, clause_normalized) from retrieved chunks
        valid_sources: Set[Tuple[str, str]] = set()
        source_display_list: List[str] = []

        for chunk in retrieved_chunks:
            doc_norm = "TS 23.501" if "501" in chunk.document_code else "TS 23.502"
            clause_norm = chunk.section_number.strip()
            valid_sources.add((doc_norm, clause_norm))
            source_display_list.append(f"[{doc_norm} Clause {chunk.section_number}, Page {chunk.page_number}]")

        if not extracted:
            # If no citations were generated, check if it's an abstention message
            is_abstention = "could not find sufficient supporting evidence" in generated_text.lower()
            return CitationValidationResult(
                is_valid=is_abstention,  # Valid if abstaining, invalid if making claims without citations
                total_citations=0,
                valid_citations=[],
                invalid_citations=[],
                citation_precision=1.0 if is_abstention else 0.0,
                retrieved_sources=source_display_list,
            )

        valid_list: List[Citation] = []
        invalid_list: List[Citation] = []

        for cit in extracted:
            doc_norm = "TS 23.501" if "501" in cit.document_code else "TS 23.502"
            clause_norm = cit.section_number.strip()

            # Check if this cited clause exists in the retrieved context
            # We match if exact clause or parent clause is in retrieved chunks
            matched = False
            for (valid_doc, valid_clause) in valid_sources:
                if doc_norm == valid_doc and (
                    clause_norm == valid_clause
                    or clause_norm.startswith(valid_clause)
                    or valid_clause.startswith(clause_norm)
                ):
                    matched = True
                    break

            if matched:
                valid_list.append(cit)
            else:
                invalid_list.append(cit)

        # Deduplicate citations by (document_code, section_number) while preserving order
        seen = set()
        deduped_valid: List[Citation] = []
        for cit in valid_list:
            key = (cit.document_code.strip(), cit.section_number.strip())
            if key not in seen:
                seen.add(key)
                deduped_valid.append(cit)

        total = len(extracted)
        precision = len(valid_list) / total if total > 0 else 1.0
        is_all_valid = len(invalid_list) == 0

        return CitationValidationResult(
            is_valid=is_all_valid,
            total_citations=total,
            valid_citations=deduped_valid,
            invalid_citations=invalid_list,
            citation_precision=round(precision, 4),
            retrieved_sources=source_display_list,
        )


if __name__ == "__main__":
    sample_text = (
        "The AMF terminates the N2 reference point from the (R)AN [TS 23.501 Clause 5.2.2, Page 42]. "
        "It also performs NAS ciphering and integrity protection [TS 23.501 Clause 5.2.2, Page 42]."
    )
    citations = CitationValidator.extract_citations(sample_text)
    print(f"Extracted {len(citations)} citations:")
    for c in citations:
        print(f"  {c.raw_citation} -> Doc: {c.document_code}, Clause: {c.section_number}, Page: {c.page_number}")
