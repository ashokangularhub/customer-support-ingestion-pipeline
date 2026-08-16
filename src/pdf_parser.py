"""PDF structural parsing: separates narrative text from tables, in reading
order, and captures the nearest heading for each block.

Uses pdfplumber (not pypdf) because it exposes per-page table bounding boxes,
which is what lets us tell a "table" chunk apart from a "text" chunk and crop
around tables to avoid duplicating their content in narrative chunks.
"""
import re
from dataclasses import dataclass, field

import pdfplumber

from config import DOC_VERSION_PATTERN

HEADING_PATTERN = re.compile(r"^\d+\.\s+\S")


@dataclass
class TableBlock:
    page_number: int
    table_index: int
    rows: list[list[str]]
    heading: str | None
    section_heading: str | None


@dataclass
class TextBlock:
    page_number: int
    text: str
    section_heading: str | None


@dataclass
class ParsedDocument:
    total_pages: int
    document_version: str | None
    last_review: str | None
    blocks: list[TextBlock | TableBlock] = field(default_factory=list)


def _clean_row(row: list[str | None]) -> list[str]:
    return [(cell or "").strip().replace("\n", " ") for cell in row]


def _last_nonblank_line(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _looks_like_caption(line: str | None) -> bool:
    """Heuristic: a table caption is short and doesn't read like prose.

    Filters out cases where the nearest line above a table is actually the
    tail of a narrative sentence rather than a heading/label (e.g. "...
    except for the wallet-credit fast-track option described below.").
    """
    if not line or len(line) > 60 or line.endswith("."):
        return False
    return len(line.split()) <= 8


def _update_section_heading(text: str, current: str | None) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if HEADING_PATTERN.match(stripped):
            current = stripped
    return current


def _extract_band_text(page, top: float, bottom: float) -> str:
    if bottom <= top:
        return ""
    cropped = page.crop((0, top, page.width, bottom))
    return cropped.extract_text() or ""


def parse_pdf(file_path: str) -> ParsedDocument:
    doc = ParsedDocument(
        total_pages=0, document_version=None, last_review=None)

    with pdfplumber.open(file_path) as pdf:
        doc.total_pages = len(pdf.pages)
        section_heading: str | None = None

        for page_index, page in enumerate(pdf.pages):
            page_number = page_index + 1

            if page_number == 1:
                first_page_text = page.extract_text() or ""
                match = DOC_VERSION_PATTERN.search(first_page_text)
                if match:
                    doc.document_version = match.group("version")
                    doc.last_review = match.group("review_date")

            tables = sorted(page.find_tables(), key=lambda t: t.bbox[1])
            table_index = 0
            y_cursor = 0.0

            for table in tables:
                top, bottom = table.bbox[1], table.bbox[3]
                band_text = _extract_band_text(page, y_cursor, top)
                candidate_heading = _last_nonblank_line(band_text)
                heading_for_table = (
                    candidate_heading if _looks_like_caption(
                        candidate_heading) else None
                )
                section_heading = _update_section_heading(
                    band_text, section_heading)

                if band_text.strip():
                    doc.blocks.append(TextBlock(
                        page_number=page_number,
                        text=band_text.strip(),
                        section_heading=section_heading,
                    ))

                rows = [_clean_row(r) for r in table.extract() or []]
                rows = [r for r in rows if any(cell for cell in r)]
                if rows:
                    doc.blocks.append(TableBlock(
                        page_number=page_number,
                        table_index=table_index,
                        rows=rows,
                        heading=heading_for_table,
                        section_heading=section_heading,
                    ))
                    table_index += 1

                y_cursor = bottom

            trailing_text = _extract_band_text(page, y_cursor, page.height)
            section_heading = _update_section_heading(
                trailing_text, section_heading)
            if trailing_text.strip():
                doc.blocks.append(TextBlock(
                    page_number=page_number,
                    text=trailing_text.strip(),
                    section_heading=section_heading,
                ))

    return doc
