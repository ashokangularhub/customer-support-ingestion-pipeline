"""Turns a ParsedDocument (text/table blocks) into LangChain Documents with
rich, production-style metadata: doc type, page, content type (text/table),
table name, section heading, product tagging, and content hashing.
"""
import hashlib
import os
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from weaviate.util import generate_uuid5

from config import (
    KNOWN_PRODUCTS,
    PRODUCT_NAME_PATTERN,
    TABLE_MAX_CHARS,
    TEXT_CHUNK_OVERLAP,
    TEXT_CHUNK_SIZE,
    detect_doc_type,
)
from pdf_parser import ParsedDocument, TableBlock, TextBlock, parse_pdf

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=TEXT_CHUNK_SIZE,
    chunk_overlap=TEXT_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _find_product(*texts: str) -> tuple[str | None, str | None]:
    for text in texts:
        if not text:
            continue
        match = PRODUCT_NAME_PATTERN.search(text)
        if match:
            name = match.group(0)
            product = next(
                p for p in KNOWN_PRODUCTS if p["product_name"] == name)
            return product["product_name"], product["product_id"]
    return None, None


def _rows_to_markdown(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[:len(header)]) + " |")
    return "\n".join(lines)


def _table_chunks_text(table: TableBlock) -> list[str]:
    """Serializes a table to markdown, splitting into row-groups (each
    repeating the header row) if the whole table would exceed TABLE_MAX_CHARS.
    """
    header, *data_rows = table.rows
    full_text = _rows_to_markdown(header, data_rows)
    if len(full_text) <= TABLE_MAX_CHARS or not data_rows:
        return [full_text]

    chunks = []
    group: list[list[str]] = []
    group_len = len(" | ".join(header))
    for row in data_rows:
        row_len = len(" | ".join(row))
        if group and group_len + row_len > TABLE_MAX_CHARS:
            chunks.append(_rows_to_markdown(header, group))
            group, group_len = [], len(" | ".join(header))
        group.append(row)
        group_len += row_len
    if group:
        chunks.append(_rows_to_markdown(header, group))
    return chunks


def compute_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_documents(file_path: str) -> list[Document]:
    file_name = os.path.basename(file_path)
    doc_type, doc_title = detect_doc_type(file_name)
    parsed: ParsedDocument = parse_pdf(file_path)
    file_hash = compute_file_hash(file_path)

    base_metadata = {
        "source_file": file_name,
        "doc_type": doc_type,
        "doc_title": doc_title,
        "total_pages": parsed.total_pages,
        "document_version": parsed.document_version or "",
        "last_review": parsed.last_review or "",
        "file_hash": file_hash,
    }

    documents: list[Document] = []
    ingested_at = datetime.now(timezone.utc).isoformat()

    for block in parsed.blocks:
        if isinstance(block, TextBlock):
            product_name, product_id = _find_product(
                block.section_heading or "", block.text)
            for piece in _text_splitter.split_text(block.text):
                if not piece.strip():
                    continue
                metadata = {
                    **base_metadata,
                    "page_number": block.page_number,
                    "content_type": "text",
                    "section_heading": block.section_heading or "",
                    "table_name": "",
                    "table_index": -1,
                    "product_name": product_name or "",
                    "product_id": product_id or "",
                }
                documents.append(
                    Document(page_content=piece, metadata=metadata))

        elif isinstance(block, TableBlock):
            heading = block.heading or block.section_heading or ""
            table_name = f"{heading} (p.{block.page_number})" if heading else \
                f"{doc_title} table {block.table_index} (p.{block.page_number})"
            product_name, product_id = _find_product(
                block.section_heading or "", heading)
            for piece in _table_chunks_text(block):
                metadata = {
                    **base_metadata,
                    "page_number": block.page_number,
                    "content_type": "table",
                    "section_heading": block.section_heading or "",
                    "table_name": table_name,
                    "table_index": block.table_index,
                    "product_name": product_name or "",
                    "product_id": product_id or "",
                }
                documents.append(
                    Document(page_content=piece, metadata=metadata))

    total_chunks = len(documents)
    for i, document in enumerate(documents):
        document.metadata["chunk_index"] = i
        document.metadata["total_chunks"] = total_chunks
        document.metadata["char_count"] = len(document.page_content)
        document.metadata["ingested_at"] = ingested_at
        content_hash = hashlib.sha256(
            document.page_content.encode("utf-8")).hexdigest()
        document.metadata["content_hash"] = content_hash
        document.metadata["chunk_id"] = generate_uuid5(
            content_hash, namespace=file_path)

    return documents
