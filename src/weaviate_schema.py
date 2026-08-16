"""Explicit Weaviate collection schema for the Aurora RAG pipeline.

We deliberately define the schema by hand instead of relying on Weaviate's
auto-schema so that:
  1. Vectors are always supplied by the client (vectorizer "none") — the
     pipeline is fully in control of embeddings, not a server-side module.
  2. Text fields get the right tokenization/BM25 config for real keyword
     search, instead of whatever auto-schema happens to infer.
  3. Numeric/date metadata (page_number, chunk_index, ingested_at, ...) get
     correct data types so filtering and aggregation behave predictably.
"""
from config import ALL_COLLECTIONS

TEXT_KEY = "text"

_KEYWORD_TEXT_PROPS = [
    "source_file", "doc_type", "doc_title", "content_type",
    "table_name", "product_name", "product_id",
    "document_version", "last_review", "content_hash", "chunk_id",
    "file_hash",
]

_SEARCHABLE_TEXT_PROPS = ["section_heading"]

_INT_PROPS = ["page_number", "total_pages",
              "table_index", "chunk_index", "total_chunks", "char_count"]


def build_schema(index_name: str) -> dict:
    properties = [
        {
            "name": TEXT_KEY,
            "dataType": ["text"],
            "tokenization": "word",
            "indexSearchable": True,
            "indexFilterable": True,
        }
    ]

    for name in _KEYWORD_TEXT_PROPS:
        properties.append({
            "name": name,
            "dataType": ["text"],
            "tokenization": "field",
            "indexSearchable": True,
            "indexFilterable": True,
        })

    for name in _SEARCHABLE_TEXT_PROPS:
        properties.append({
            "name": name,
            "dataType": ["text"],
            "tokenization": "word",
            "indexSearchable": True,
            "indexFilterable": True,
        })

    for name in _INT_PROPS:
        properties.append({
            "name": name,
            "dataType": ["int"],
            "indexFilterable": True,
            "indexRangeFilters": True,
        })

    properties.append({
        "name": "ingested_at",
        "dataType": ["date"],
        "indexFilterable": True,
    })

    return {
        "class": index_name,
        "vectorizer": "none",
        "invertedIndexConfig": {
            "bm25": {"b": 0.75, "k1": 1.2},
        },
        "properties": properties,
    }


def reset_collection(client, index_name: str) -> None:
    """Deletes the collection if it exists (used for a clean re-ingestion)."""
    if client.collections.exists(index_name):
        client.collections.delete(index_name)


def ensure_collection(client, index_name: str) -> None:
    if not client.collections.exists(index_name):
        client.collections.create_from_dict(build_schema(index_name))


def reset_all_collections(client) -> None:
    for index_name in ALL_COLLECTIONS:
        reset_collection(client, index_name)


def ensure_all_collections(client) -> None:
    for index_name in ALL_COLLECTIONS:
        ensure_collection(client, index_name)
