"""Clause-aware PDF ingestion and text extraction for 3GPP specifications."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pymupdf

from src.models import ChunkMetadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DOC_METADATA: Dict[str, Dict[str, str]] = {
    "ts_123501v170400p.pdf": {
        "document_code": "3GPP TS 23.501",
        "document_title": "System architecture for the 5G System (5GS)",
        "release": "17",
        "version": "17.4.0",
    },
    "ts_123502v160900p.pdf": {
        "document_code": "3GPP TS 23.502",
        "document_title": "Procedures for the 5G System (5GS)",
        "release": "16",
        "version": "16.9.0",
    },
}

TOC_DOTS_PATTERN = re.compile(r"\.{3,}\s*\d+$")

# Regex to detect 3GPP numbered headings: e.g. "5.2.2 Access and Mobility Management Function (AMF)"
HEADING_SINGLE_LINE = re.compile(r"^(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z0-9\(\)\/\-\s,:'\"]+)$", re.MULTILINE)
ANNEX_HEADING = re.compile(r"^(Annex\s+[A-Z](?:\s*\(.*?\))?):?\s*([^\n\r]*)$", re.MULTILINE)

HEADER_PATTERNS = [
    re.compile(r"^ETSI\s*$", re.IGNORECASE),
    re.compile(r"^ETSI\s+TS\s+123\s+\d+\s+V\d+\.\d+\.\d+.*$", re.IGNORECASE),
    re.compile(r"^3GPP\s+TS\s+23\.\d+\s+version\s+\d+\.\d+\.\d+\s+Release\s+\d+.*$", re.IGNORECASE),
    re.compile(r"^\d+\s*$", re.MULTILINE),  # Standalone page numbers
]


class ExtractedSection:
    """Represents an extracted clause section spanning one or more pages."""

    def __init__(
        self,
        document_code: str,
        document_title: str,
        release: str,
        version: str,
        section_number: str,
        section_title: str,
        start_page: int,
    ):
        self.document_code = document_code
        self.document_title = document_title
        self.release = release
        self.version = version
        self.section_number = section_number
        self.section_title = section_title
        self.start_page = start_page
        self.pages: List[int] = [start_page]
        self.paragraphs: List[Tuple[int, str]] = []  # (page_number, text)


def clean_page_text(raw_text: str) -> str:
    """Strip repeated running headers, footers, standalone page digits, and noise."""
    lines = raw_text.splitlines()
    cleaned_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check if line matches running header / footer noise
        is_noise = False
        for pat in HEADER_PATTERNS:
            if pat.match(stripped):
                is_noise = True
                break

        if not is_noise:
            cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def is_table_of_contents_page(raw_text: str) -> bool:
    """Determine if a page belongs to the Table of Contents."""
    lines = raw_text.splitlines()
    dot_leader_count = sum(1 for line in lines if TOC_DOTS_PATTERN.search(line.strip()))
    return dot_leader_count >= 2


def extract_document_sections(pdf_path: Path) -> List[ExtractedSection]:
    """Parse a 3GPP PDF into structured, clause-delimited sections."""
    filename = pdf_path.name
    meta = DOC_METADATA.get(
        filename,
        {
            "document_code": "3GPP TS 23.501" if "23501" in filename else "3GPP TS 23.502",
            "document_title": "5G System Specification",
            "release": "17",
            "version": "17.0.0",
        },
    )

    logger.info(f"Opening PDF: {pdf_path} ({meta['document_code']} Rel-{meta['release']} v{meta['version']})")
    doc = pymupdf.open(str(pdf_path))
    total_pages = len(doc)

    sections: List[ExtractedSection] = []
    current_section: Optional[ExtractedSection] = None
    passed_toc = False

    for page_idx in range(total_pages):
        page_num = page_idx + 1  # 1-based page number
        raw_text = doc[page_idx].get_text("text")

        # Skip Table of Contents and Cover front matter
        if not passed_toc:
            if is_table_of_contents_page(raw_text) or page_num <= 18:
                continue
            # Check if this page starts real content (Foreword or 1 Scope with no TOC dot leaders)
            if ("Foreword" in raw_text or "1\nScope" in raw_text or "1 \nScope" in raw_text or "1 Scope" in raw_text) and not is_table_of_contents_page(raw_text):
                passed_toc = True
            else:
                continue

        cleaned_text = clean_page_text(raw_text)
        if not cleaned_text.strip():
            continue

        lines = cleaned_text.splitlines()
        i = 0
        current_paragraph_buffer: List[str] = []

        def flush_buffer():
            nonlocal current_paragraph_buffer, current_section
            if current_paragraph_buffer and current_section:
                para_text = "\n".join(current_paragraph_buffer).strip()
                if para_text:
                    current_section.paragraphs.append((page_num, para_text))
                current_paragraph_buffer = []

        while i < len(lines):
            line = lines[i].strip()

            # Check for Annex headings
            annex_match = ANNEX_HEADING.match(line)
            if annex_match:
                flush_buffer()
                sec_num = annex_match.group(1).strip()
                sec_title = annex_match.group(2).strip() or "Annex"
                current_section = ExtractedSection(
                    document_code=meta["document_code"],
                    document_title=meta["document_title"],
                    release=meta["release"],
                    version=meta["version"],
                    section_number=sec_num,
                    section_title=sec_title,
                    start_page=page_num,
                )
                sections.append(current_section)
                i += 1
                continue

            # Check for two-line heading: "5.2.2" followed by "Access and Mobility..."
            if re.match(r"^(\d+(?:\.\d+)*)$", line) and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if re.match(r"^[A-Z][A-Za-z0-9\(\)\/\-\s,:'\"]{2,}$", next_line) and not next_line.endswith("."):
                    flush_buffer()
                    sec_num = line
                    sec_title = next_line
                    current_section = ExtractedSection(
                        document_code=meta["document_code"],
                        document_title=meta["document_title"],
                        release=meta["release"],
                        version=meta["version"],
                        section_number=sec_num,
                        section_title=sec_title,
                        start_page=page_num,
                    )
                    sections.append(current_section)
                    i += 2
                    continue

            # Check for single-line heading: "5.2.2 Access and Mobility..."
            single_match = HEADING_SINGLE_LINE.match(line)
            if single_match:
                sec_num = single_match.group(1).strip()
                sec_title = single_match.group(2).strip()
                if len(sec_title) < 120 and not sec_title.endswith("."):
                    flush_buffer()
                    current_section = ExtractedSection(
                        document_code=meta["document_code"],
                        document_title=meta["document_title"],
                        release=meta["release"],
                        version=meta["version"],
                        section_number=sec_num,
                        section_title=sec_title,
                        start_page=page_num,
                    )
                    sections.append(current_section)
                    i += 1
                    continue

            # Check for Foreword
            if line == "Foreword":
                flush_buffer()
                current_section = ExtractedSection(
                    document_code=meta["document_code"],
                    document_title=meta["document_title"],
                    release=meta["release"],
                    version=meta["version"],
                    section_number="Foreword",
                    section_title="Foreword",
                    start_page=page_num,
                )
                sections.append(current_section)
                i += 1
                continue

            if current_section is None:
                current_section = ExtractedSection(
                    document_code=meta["document_code"],
                    document_title=meta["document_title"],
                    release=meta["release"],
                    version=meta["version"],
                    section_number="General",
                    section_title="General",
                    start_page=page_num,
                )
                sections.append(current_section)

            if page_num not in current_section.pages:
                current_section.pages.append(page_num)

            current_paragraph_buffer.append(line)
            i += 1

        flush_buffer()

    doc.close()
    logger.info(f"Extracted {len(sections)} distinct clause sections from {filename}")
    return sections
