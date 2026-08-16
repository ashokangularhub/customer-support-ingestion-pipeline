import os

import weaviate
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import EMBEDDING_MODEL, INDEX_NAME

load_dotenv()

# Hybrid alpha: 0.0 = pure BM25 keyword, 1.0 = pure vector, in between = blended.
HYBRID_ALPHA = 0.75

BM25_QUERY_PROPERTIES = ["text", "section_heading",
                         "table_name", "product_name"]

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the "
    "provided context. If the answer isn't in the context, say you don't know."
)

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


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


def _search(collection, query: str, search_type: str, k: int, embeddings: OpenAIEmbeddings):
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
    return [_object_to_chunk(obj) for obj in search_result.objects]


def ask_question(query: str, search_type: str = "hybrid", k: int = 8) -> dict:
    client = weaviate.connect_to_local()
    try:
        collection = client.collections.get(INDEX_NAME)
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

        context_chunks = _search(collection, query, search_type, k, embeddings)
        context_text = "\n\n".join(
            f"[{c['doc_type']} | p.{c['page_number']} | {c['content_type']}"
            f"{' | ' + c['table_name'] if c['table_name'] else ''}]\n{c['text']}"
            for c in context_chunks
        )

        messages = PROMPT_TEMPLATE.format_messages(
            context=context_text, question=query)

        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        response = llm.invoke(messages)

        return {
            "system_prompt": SYSTEM_PROMPT,
            "context": context_chunks,
            "question": query,
            "search_type": search_type,
            "answer": response.content,
        }
    finally:
        client.close()


if __name__ == "__main__":
    result = ask_question(
        "What are the trouble shooting steps for  AuroraWatch Fit 3 ?")
    print(result["answer"])
