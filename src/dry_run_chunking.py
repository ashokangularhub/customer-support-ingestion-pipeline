"""Dry-run the chunking stage for a single file, with zero Weaviate I/O.

Calls chunking.build_documents() directly (parsing + splitting only) and
writes a full inspection report of every chunk's metadata and content to
both stdout and rag_chunking_debug.txt, so chunking/table-extraction quality
can be reviewed before anything is embedded or written to the vector store.

Usage:
    python dry_run_chunking.py [file_name_or_path]

Defaults to aurora_product_information_catalog.pdf in rag-files/.
"""
import argparse
import json
import os
from datetime import datetime, timezone

from chunking import build_documents
from config import INCOMING_FOLDER

LOG_FILE = "rag_chunking_debug.txt"
TEXT_PREVIEW_CHARS = 300  # table chunks are always printed in full


def resolve_path(file_arg: str) -> str:
    if os.path.isfile(file_arg):
        return file_arg
    candidate = os.path.join(INCOMING_FOLDER, file_arg)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"Could not find file: {file_arg}")


class Tee:
    """Writes every line to both stdout and the debug log file."""

    def __init__(self, log_path: str):
        self._fh = open(log_path, "w", encoding="utf-8")

    def line(self, text: str = "") -> None:
        print(text)
        self._fh.write(text + "\n")

    def close(self) -> None:
        self._fh.close()


def _table_breakdown(table_docs: list) -> list[tuple[int, int, str, int]]:
    # table_index is only unique within a page, so group by (page, index).
    keys = sorted({(d.metadata["page_number"], d.metadata["table_index"])
                   for d in table_docs})
    breakdown = []
    for page_number, idx in keys:
        chunks_for_table = [
            d for d in table_docs
            if d.metadata["page_number"] == page_number
            and d.metadata["table_index"] == idx]
        breakdown.append(
            (page_number, idx, chunks_for_table[0].metadata["table_name"], len(chunks_for_table)))
    return breakdown


def run(file_path: str) -> None:
    tee = Tee(LOG_FILE)
    try:
        tee.line("=" * 88)
        tee.line("RAG CHUNKING DRY-RUN — no Weaviate connection, no vectors written")
        tee.line(f"File     : {file_path}")
        tee.line(f"Run at   : {datetime.now(timezone.utc).isoformat()}")
        tee.line("=" * 88)

        docs = build_documents(file_path)
        table_docs = [d for d in docs if d.metadata["content_type"] == "table"]
        text_docs = [d for d in docs if d.metadata["content_type"] == "text"]

        tee.line("")
        tee.line("SUMMARY")
        tee.line("-" * 88)
        tee.line(f"Total Chunks Generated : {len(docs)}")
        tee.line(f"Total Text Chunks      : {len(text_docs)}")
        tee.line(f"Total Table Chunks     : {len(table_docs)}")

        table_breakdown = _table_breakdown(table_docs)
        tee.line(f"Distinct Source Tables : {len(table_breakdown)}")
        for page_number, idx, name, chunk_count in table_breakdown:
            tee.line(
                f"  - page={page_number}, table_index={idx} ({name!r}) -> split into {chunk_count} chunk(s)")

        tee.line("")
        tee.line("CHUNK BREAKDOWN")
        tee.line("=" * 88)

        for i, doc in enumerate(docs):
            meta = doc.metadata
            is_table = meta["content_type"] == "table"
            tee.line("")
            tee.line(f"--- Chunk #{i} (chunk_index={meta['chunk_index']}) ---")
            tee.line(
                f"Chunk Type: {'TABLE [tabular content]' if is_table else 'TEXT'}")
            tee.line("Metadata:")
            tee.line(json.dumps(meta, indent=2, default=str))

            content = doc.page_content
            if is_table:
                tee.line("Content (raw markdown table, full):")
            elif len(content) > TEXT_PREVIEW_CHARS:
                total_chars = len(content)
                content = content[:TEXT_PREVIEW_CHARS] + \
                    f"... [truncated, {total_chars} chars total]"
                tee.line("Content Preview:")
            else:
                tee.line("Content (full):")
            tee.line(content)

        tee.line("")
        tee.line("=" * 88)
        tee.line("END OF DRY RUN")
    finally:
        tee.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dry-run RAG chunking for a single file (no DB writes)")
    parser.add_argument(
        "file", nargs="?", default="aurora_product_information_catalog.pdf",
        help="File name (looked up in rag-files/) or a full path")
    args = parser.parse_args()

    resolved_path = resolve_path(args.file)
    run(resolved_path)
    print(f"\nDebug log written to: {os.path.abspath(LOG_FILE)}")
