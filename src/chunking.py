"""Clause-bounded semantic chunker for 3GPP specifications."""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from config import settings
from src.ingestion import ExtractedSection, extract_document_sections
from src.models import Chunk, ChunkMetadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def sanitize_id(text: str) -> str:
    """Convert text into clean, alphanumeric identifier component."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", text).strip("_").lower()


def chunk_section(
    section: ExtractedSection,
    target_words: int = 350,
    overlap_words: int = 50,
) -> List[Chunk]:
    """Break an extracted section into clause-bounded, token/word-sized chunks."""
    chunks: List[Chunk] = []
    if not section.paragraphs:
        return chunks

    doc_short = "ts23501" if "23.501" in section.document_code else "ts23502"
    sec_id = sanitize_id(section.section_number)
    rel_id = f"r{section.release}"

    # Group paragraphs while preserving page tracking
    current_words: List[str] = []
    current_text_blocks: List[str] = []
    chunk_page = section.start_page
    chunk_counter = 1

    for page_num, para_text in section.paragraphs:
        words = para_text.split()
        if not words:
            continue

        # If adding this paragraph exceeds target word limit and we already have sufficient content
        if len(current_words) + len(words) > target_words and len(current_words) >= 150:
            full_chunk_text = "\n\n".join(current_text_blocks).strip()
            chunk_id = f"{doc_short}_{rel_id}_{sec_id}_p{chunk_page}_{chunk_counter:03d}"

            meta = ChunkMetadata(
                chunk_id=chunk_id,
                document_code=section.document_code,
                document_title=section.document_title,
                release=section.release,
                version=section.version,
                section_number=section.section_number,
                section_title=section.section_title,
                page_number=chunk_page,
                content_type="procedure_step" if any(kw in section.section_title.lower() for kw in ["procedure", "flow", "step", "registration", "session"]) else "paragraph",
            )

            chunk = Chunk(
                metadata=meta,
                text=full_chunk_text,
                enriched_text="",
            )
            chunk.enriched_text = chunk.to_enriched_string()
            chunks.append(chunk)

            # Apply sliding overlap
            chunk_counter += 1
            if len(current_words) > overlap_words:
                overlap_text = " ".join(current_words[-overlap_words:])
                current_words = overlap_text.split()
                current_text_blocks = [overlap_text]
            else:
                current_words = []
                current_text_blocks = []

        current_words.extend(words)
        current_text_blocks.append(para_text)
        chunk_page = page_num

    # Flush remaining text
    if current_text_blocks:
        full_chunk_text = "\n\n".join(current_text_blocks).strip()
        chunk_id = f"{doc_short}_{rel_id}_{sec_id}_p{chunk_page}_{chunk_counter:03d}"

        meta = ChunkMetadata(
            chunk_id=chunk_id,
            document_code=section.document_code,
            document_title=section.document_title,
            release=section.release,
            version=section.version,
            section_number=section.section_number,
            section_title=section.section_title,
            page_number=chunk_page,
            content_type="paragraph",
        )

        chunk = Chunk(
            metadata=meta,
            text=full_chunk_text,
            enriched_text="",
        )
        chunk.enriched_text = chunk.to_enriched_string()
        chunks.append(chunk)

    return chunks


def process_and_save_chunks(
    pdf_paths: Optional[List[Path]] = None,
    output_path: Optional[Path] = None,
) -> List[Chunk]:
    """Ingest all raw PDFs and persist clause-bounded chunks into JSONL."""
    if pdf_paths is None:
        pdf_paths = [
            settings.RAW_DATA_DIR / "ts_123501v170400p.pdf",
            settings.RAW_DATA_DIR / "ts_123502v160900p.pdf",
        ]

    if output_path is None:
        output_path = settings.PROCESSED_DATA_DIR / "chunks.jsonl"

    all_chunks: List[Chunk] = []

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            logger.warning(f"File not found: {pdf_path}. Skipping.")
            continue

        sections = extract_document_sections(pdf_path)
        for section in sections:
            section_chunks = chunk_section(section)
            all_chunks.extend(section_chunks)

    # Save to JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk.model_dump_json() + "\n")

    logger.info(f"Successfully processed {len(all_chunks)} chunks to {output_path}")
    return all_chunks


if __name__ == "__main__":
    process_and_save_chunks()
