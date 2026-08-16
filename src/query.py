import weaviate
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from config import ALL_COLLECTIONS, DEFAULT_TOP_K, EMBEDDING_MODEL

load_dotenv()

# Hybrid alpha: 0.0 = pure BM25 keyword, 1.0 = pure vector, in between = blended.
HYBRID_ALPHA = 0.75

BM25_QUERY_PROPERTIES = ["text", "section_heading",
                         "table_name", "product_name"]


def _object_to_chunk(obj) -> dict:
    props = obj.properties
    return {
        "text": props.get("text"),
        "source_file": props.get("source_file"),
        "doc_type": props.get("doc_type"),
        "page_number": props.get("page_number"),
        "content_type": props.get("content_type"),
        "table_name": props.get("table_name"),
        "section_heading": props.get("section_heading"),
        "product_name": props.get("product_name"),
    }


def _run_search(collection, query: str, search_type: str, k: int, embeddings: OpenAIEmbeddings):
    if search_type == "keyword":
        search_result = collection.query.bm25(
            query=query, query_properties=BM25_QUERY_PROPERTIES, limit=k)
    elif search_type == "vector":
        vector = embeddings.embed_query(query)
        search_result = collection.query.near_vector(
            near_vector=vector, limit=k)
    elif search_type == "hybrid":
        vector = embeddings.embed_query(query)
        search_result = collection.query.hybrid(
            query=query, vector=vector, alpha=HYBRID_ALPHA,
            query_properties=BM25_QUERY_PROPERTIES, limit=k)
    else:
        raise ValueError(
            f"search_type must be one of 'keyword', 'vector', 'hybrid', got {search_type!r}")
    return [_object_to_chunk(o) for o in search_result.objects]


def search(query: str, collection: str, search_type: str = "hybrid", k: int = DEFAULT_TOP_K) -> list[dict]:
    """Retrieval primitive used by the /retrieve endpoint (and by rag-retrieval
    over HTTP). Owns the Weaviate connection and query embedding so that the
    embedding model always matches what was used at ingestion time.

    `collection` is required and must be one of config.ALL_COLLECTIONS —
    there's no query-routing agent yet, so the caller must say which domain
    to search.
    """
    if collection not in ALL_COLLECTIONS:
        raise ValueError(
            f"collection must be one of {ALL_COLLECTIONS}, got {collection!r}")

    client = weaviate.connect_to_local()
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        coll = client.collections.get(collection)
        return _run_search(coll, query, search_type, k, embeddings)
    finally:
        client.close()


if __name__ == "__main__":
    results = search(
        "What are the trouble shooting steps for AuroraWatch Fit 3?",
        collection="AURORA_TECHNICAL_SUPPORT")
    for result_chunk in results:
        print(result_chunk)
