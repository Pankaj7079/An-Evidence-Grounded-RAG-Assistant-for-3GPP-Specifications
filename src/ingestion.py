"""Authoritative Ground-Truth Ingestion & Chunking Engine for 3GPP Specifications.

Extracts text precisely bounded by official 3GPP PDF bookmarks, maps exact printed
document page numbers (visible in the specification headers), handles multi-page
ranges (e.g. '42-43'), and generates separate and merged JSONL datasets.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pymupdf

from config import settings
from src.models import Chunk, ChunkMetadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DOC_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ts_123501v170400p.pdf": {
        "document_code": "3GPP TS 23.501",
        "document_title": "System architecture for the 5G System (5GS)",
        "release": "17",
        "version": "17.4.0",
        "output_filename": "ts23501_chunks.jsonl",
        "manifest_filename": "ts23501_manifest.json",
    },
    "ts_123502v160900p.pdf": {
        "document_code": "3GPP TS 23.502",
        "document_title": "Procedures for the 5G System (5GS)",
        "release": "16",
        "version": "16.9.0",
        "output_filename": "ts23502_chunks.jsonl",
        "manifest_filename": "ts23502_manifest.json",
    },
}

HEADER_PATTERNS = [
    re.compile(r"^ETSI\s*$", re.IGNORECASE),
    re.compile(r"^ETSI\s+TS\s+1[0-9]{2}\s+[0-9]+\s+V\d+\.\d+\.\d+.*$", re.IGNORECASE),
    re.compile(r"^3GPP\s+TS\s+23\.\d+\s+version\s+\d+\.\d+\.\d+\s+Release\s+\d+.*$", re.IGNORECASE),
    re.compile(r"^\d{1,4}\s*$", re.MULTILINE),
    re.compile(r"^Release\s+\d+\s*$", re.IGNORECASE),
]

TARGET_CHARS = 1400
MAX_CHARS = 2400
MIN_CHARS = 60

PROCEDURE_KEYWORDS = frozenset(
    [
        "procedure",
        "registration",
        "session",
        "handover",
        "deregistration",
        "authentication",
        "flow",
        "step",
        "request",
        "response",
        "establishment",
        "modification",
    ]
)


def extract_printed_page_number(raw_text: str, pdf_page: int) -> int:
    """Extract printed page number visible in the 3GPP document header."""
    lines = [l.strip() for l in raw_text.splitlines()[:10] if l.strip()]
    for i, l in enumerate(lines):
        if re.match(r"^\d{1,4}$", l):
            if (i > 0 and "ETSI" in lines[i - 1]) or (i + 1 < len(lines) and "3GPP" in lines[i + 1]):
                return int(l)
    return max(1, pdf_page - 1)


def clean_page_lines(raw_text: str) -> List[str]:
    """Clean running headers, footers, and page numbers from text lines."""
    cleaned = []
    for line in raw_text.splitlines():
        s = line.strip()
        if not s:
            continue
        if any(pat.match(s) for pat in HEADER_PATTERNS):
            continue
        cleaned.append(s)
    return cleaned


def parse_toc_entry(title: str) -> Tuple[str, str]:
    """Extract (section_number, section_title) from TOC title."""
    title = title.strip()
    m = re.match(r"^(\d+(?:\.\d+)*[a-z]?)\s+(.*)$", title)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    m = re.match(r"^(Annex\s+[A-Z](?:\s*\([^)]+\))?):?\s*(.*)$", title, re.I)
    if m:
        sec_num = m.group(1).strip()
        sec_title = m.group(2).strip() or sec_num
        return sec_num, sec_title

    return title, title


def get_ground_truth_clauses(doc: pymupdf.Document) -> List[Dict[str, Any]]:
    """Extract full technical clause tree from PDF bookmarks."""
    raw_toc = doc.get_toc()
    clauses = []
    level_stack: Dict[int, str] = {}

    for level, title, pdf_page in raw_toc:
        t_lower = title.lower().strip()
        if t_lower in [
            "intellectual property rights",
            "legal notice",
            "modal verbs terminology",
            "contents",
        ]:
            continue
        if "change history" in t_lower:
            continue

        sec_num, sec_title = parse_toc_entry(title)

        level_stack[level] = f"{sec_num} {sec_title}".strip()
        for l in list(level_stack.keys()):
            if l > level:
                del level_stack[l]

        breadcrumb = " > ".join(level_stack[l] for l in sorted(level_stack.keys()))

        raw_text = doc[pdf_page - 1].get_text("text") if pdf_page <= len(doc) else ""
        printed_page = extract_printed_page_number(raw_text, pdf_page)

        clauses.append(
            {
                "level": level,
                "section_number": sec_num,
                "section_title": sec_title,
                "section_hierarchy": breadcrumb,
                "pdf_page": pdf_page,
                "page_number": printed_page,
            }
        )

    return clauses


def extract_all_page_data(doc: pymupdf.Document) -> Tuple[Dict[int, List[str]], Dict[int, int]]:
    """Extract cleaned lines and printed page numbers for all pages."""
    pages_lines = {}
    printed_page_map = {}
    for p_idx in range(len(doc)):
        pdf_page = p_idx + 1
        raw_text = doc[p_idx].get_text("text")
        pages_lines[pdf_page] = clean_page_lines(raw_text)
        printed_page_map[pdf_page] = extract_printed_page_number(raw_text, pdf_page)
    return pages_lines, printed_page_map


def locate_heading_in_lines(lines: List[str], sec_num: str, sec_title: str) -> Optional[int]:
    """Find line index where clause heading starts."""
    clean_num = sec_num.strip()
    for idx, l in enumerate(lines):
        if l == clean_num or l.startswith(f"{clean_num} ") or l.startswith(f"{clean_num}\t"):
            return idx
        if clean_num.startswith("Annex") and l.startswith("Annex"):
            return idx
        if clean_num == "Foreword" and l == "Foreword":
            return idx
    return None


def extract_clause_texts_precise(
    doc: pymupdf.Document,
    clauses: List[Dict[str, Any]],
    pages_lines: Dict[int, List[str]],
    printed_page_map: Dict[int, int],
) -> List[Dict[str, Any]]:
    """Slice text precisely between clause headings across pages."""
    num_clauses = len(clauses)

    for i, clause in enumerate(clauses):
        start_pdf_page = clause["pdf_page"]
        next_clause = clauses[i + 1] if i + 1 < num_clauses else None
        end_pdf_page = next_clause["pdf_page"] if next_clause else len(doc)

        clause_text_blocks: List[Tuple[int, str]] = []

        if start_pdf_page == end_pdf_page:
            lines = pages_lines.get(start_pdf_page, [])
            start_idx = locate_heading_in_lines(lines, clause["section_number"], clause["section_title"]) or 0
            end_idx = len(lines)
            if next_clause:
                found_next = locate_heading_in_lines(
                    lines[start_idx + 1 :], next_clause["section_number"], next_clause["section_title"]
                )
                if found_next is not None:
                    end_idx = start_idx + 1 + found_next

            clause_lines = lines[start_idx:end_idx]
            if clause_lines:
                clause_text_blocks.append(
                    (printed_page_map[start_pdf_page], "\n".join(clause_lines))
                )
        else:
            for p in range(start_pdf_page, end_pdf_page + 1):
                lines = pages_lines.get(p, [])
                if not lines:
                    continue

                if p == start_pdf_page:
                    start_idx = locate_heading_in_lines(lines, clause["section_number"], clause["section_title"]) or 0
                    clause_lines = lines[start_idx:]
                elif p == end_pdf_page:
                    end_idx = len(lines)
                    if next_clause:
                        found_next = locate_heading_in_lines(
                            lines, next_clause["section_number"], next_clause["section_title"]
                        )
                        if found_next is not None:
                            end_idx = found_next
                    clause_lines = lines[:end_idx]
                else:
                    clause_lines = lines

                if clause_lines:
                    clause_text_blocks.append((printed_page_map[p], "\n".join(clause_lines)))

        clause["text_blocks"] = clause_text_blocks

    return clauses


def chunk_clause(meta_cfg: Dict[str, Any], clause: Dict[str, Any]) -> List[Chunk]:
    """Convert an extracted clause into structured, bounded chunks."""
    chunks: List[Chunk] = []
    text_blocks = clause.get("text_blocks", [])
    if not text_blocks:
        return chunks

    doc_short = "ts23501" if "23.501" in meta_cfg["document_code"] else "ts23502"
    sec_id = re.sub(r"[^a-z0-9_]", "_", clause["section_number"].lower()).strip("_")
    if not sec_id:
        sec_id = "clause"

    is_procedure = any(kw in clause["section_title"].lower() for kw in PROCEDURE_KEYWORDS)
    content_type = "procedure_step" if is_procedure else "paragraph"

    counter = 1
    current_lines: List[Tuple[int, str]] = []
    current_chars = 0

    def flush():
        nonlocal current_lines, current_chars, counter
        if not current_lines:
            return

        text = "\n".join(l[1] for l in current_lines).strip()
        if len(text) >= MIN_CHARS:
            start_p = current_lines[0][0]
            end_p = current_lines[-1][0]
            page_display = f"{start_p}" if start_p == end_p else f"{start_p}-{end_p}"
            page_id_str = f"p{start_p}" if start_p == end_p else f"p{start_p}_{end_p}"

            chunk_id = f"{doc_short}_r{meta_cfg['release']}_{sec_id}_{page_id_str}_{counter:03d}"
            meta = ChunkMetadata(
                chunk_id=chunk_id,
                document_code=meta_cfg["document_code"],
                document_title=meta_cfg["document_title"],
                release=meta_cfg["release"],
                version=meta_cfg["version"],
                section_number=clause["section_number"],
                section_title=clause["section_title"],
                section_hierarchy=clause["section_hierarchy"],
                start_page=start_p,
                end_page=end_p,
                page_number=page_display,
                content_type=content_type,
            )
            c = Chunk(metadata=meta, text=text, enriched_text="")
            c.enriched_text = c.to_enriched_string()
            chunks.append(c)
            counter += 1
        current_lines.clear()
        current_chars = 0

    for p_num, block_text in text_blocks:
        lines = block_text.splitlines()
        for line in lines:
            line_len = len(line) + 1
            if current_chars + line_len > MAX_CHARS and current_lines:
                flush()

            current_lines.append((p_num, line))
            current_chars += line_len

    flush()
    return chunks


def parse_specification(pdf_filename: str) -> Tuple[List[Chunk], Dict[str, Any]]:
    """Parse one 3GPP PDF into its dedicated JSONL file and manifest."""
    cfg = DOC_CONFIGS[pdf_filename]
    pdf_path = settings.RAW_DATA_DIR / pdf_filename

    logger.info(f"Opening: {pdf_filename} ({cfg['document_code']} Rel-{cfg['release']})")
    doc = pymupdf.open(str(pdf_path))

    clauses = get_ground_truth_clauses(doc)
    logger.info(f"Found {len(clauses)} ground-truth clauses in {cfg['document_code']}")

    pages_lines, printed_page_map = extract_all_page_data(doc)
    extract_clause_texts_precise(doc, clauses, pages_lines, printed_page_map)

    all_chunks: List[Chunk] = []
    clause_manifest = []

    for clause in clauses:
        c_chunks = chunk_clause(cfg, clause)
        all_chunks.extend(c_chunks)
        clause_manifest.append(
            {
                "section_number": clause["section_number"],
                "section_title": clause["section_title"],
                "page_number": clause["page_number"],
                "hierarchy": clause["section_hierarchy"],
                "chunks_count": len(c_chunks),
            }
        )

    out_file = settings.PROCESSED_DATA_DIR / cfg["output_filename"]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk.model_dump_json(exclude_none=True) + "\n")

    manifest_file = settings.PROCESSED_DATA_DIR / cfg["manifest_filename"]
    manifest_data = {
        "document_code": cfg["document_code"],
        "title": cfg["document_title"],
        "release": cfg["release"],
        "version": cfg["version"],
        "total_clauses": len(clauses),
        "total_chunks": len(all_chunks),
        "clauses": clause_manifest,
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    doc.close()
    logger.info(
        f"Generated {len(all_chunks)} chunks for {cfg['document_code']} -> {out_file}"
    )
    return all_chunks, manifest_data


def run_full_pipeline():
    """Execute complete ingestion pipeline generating separate and merged JSONL files."""
    chunks_501, _ = parse_specification("ts_123501v170400p.pdf")
    chunks_502, _ = parse_specification("ts_123502v160900p.pdf")

    merged_file = settings.PROCESSED_DATA_DIR / "chunks.jsonl"
    all_chunks = chunks_501 + chunks_502
    with open(merged_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk.model_dump_json(exclude_none=True) + "\n")

    logger.info(
        f"Master dataset generated: {len(all_chunks)} total chunks -> {merged_file}"
    )


if __name__ == "__main__":
    run_full_pipeline()
