"""Read-only browsing API over Weaviate collections, used by the static admin UI.

Each request opens and closes its own client connection, matching the pattern
already used throughout this project (ingestion.py, query.py, inspect_data.py)
instead of sharing a single long-lived client across concurrent requests.
"""
from typing import Optional

import weaviate
from fastapi import APIRouter, HTTPException, Query
from weaviate.classes.query import Filter

router = APIRouter(prefix="/api", tags=["admin"])

FACET_PROPERTIES = ("doc_type", "content_type", "source_file")
BM25_QUERY_PROPERTIES = [
    "text", "section_heading", "table_name", "product_name"]


def _client():
    return weaviate.connect_to_local()


def _require_collection(client, name: str):
    if not client.collections.exists(name):
        raise HTTPException(
            status_code=404, detail=f"Collection '{name}' not found")
    return client.collections.get(name)


def _build_filter(doc_type: Optional[str], content_type: Optional[str],
                  source_file: Optional[str]):
    clauses = []
    if doc_type:
        clauses.append(Filter.by_property("doc_type").equal(doc_type))
    if content_type:
        clauses.append(Filter.by_property("content_type").equal(content_type))
    if source_file:
        clauses.append(Filter.by_property("source_file").equal(source_file))
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else Filter.all_of(clauses)


def _vector_summary(obj, include_vector: bool) -> dict:
    if not include_vector or not obj.vector:
        return {}
    vector = obj.vector.get("default", obj.vector)
    return {"vector_preview": vector[:8], "vector_dim": len(vector)}


@router.get("/collections")
def list_collections():
    client = _client()
    try:
        configs = client.collections.list_all(simple=True)
        collections = []
        for name, cfg in configs.items():
            collection = client.collections.get(name)
            count = collection.aggregate.over_all(total_count=True).total_count
            collections.append({
                "name": name,
                "count": count,
                "properties": [p.name for p in cfg.properties],
            })
        return collections
    finally:
        client.close()


@router.get("/collections/{name}/facets")
def get_facets(name: str):
    client = _client()
    try:
        collection = _require_collection(client, name)
        facets = {}
        for prop in FACET_PROPERTIES:
            result = collection.aggregate.over_all(
                group_by=prop, total_count=True)
            facets[prop] = sorted(
                g.grouped_by.value for g in result.groups if g.grouped_by.value)
        return facets
    finally:
        client.close()


@router.get("/collections/{name}/objects")
def list_objects(
    name: str,
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
    doc_type: Optional[str] = None,
    content_type: Optional[str] = None,
    source_file: Optional[str] = None,
    search: Optional[str] = None,
    include_vector: bool = False,
):
    client = _client()
    try:
        collection = _require_collection(client, name)
        where = _build_filter(doc_type, content_type, source_file)

        if search:
            result = collection.query.bm25(
                query=search, filters=where, limit=limit,
                query_properties=BM25_QUERY_PROPERTIES,
                include_vector=include_vector,
            )
            # BM25 doesn't return a total match count; report what was fetched.
            total = len(result.objects)
        else:
            result = collection.query.fetch_objects(
                filters=where, limit=limit, offset=offset,
                include_vector=include_vector,
            )
            total = collection.aggregate.over_all(
                total_count=True, filters=where).total_count

        objects = [
            {
                "uuid": str(obj.uuid),
                "properties": obj.properties,
                **_vector_summary(obj, include_vector),
            }
            for obj in result.objects
        ]
        return {"total": total, "limit": limit, "offset": offset, "objects": objects}
    finally:
        client.close()


@router.get("/collections/{name}/objects/{uuid}")
def get_object(name: str, uuid: str):
    client = _client()
    try:
        collection = _require_collection(client, name)
        obj = collection.query.fetch_object_by_id(uuid, include_vector=True)
        if obj is None:
            raise HTTPException(status_code=404, detail="Object not found")
        vector = obj.vector.get("default", obj.vector) if obj.vector else []
        return {
            "uuid": str(obj.uuid),
            "properties": obj.properties,
            "vector": vector,
            "vector_dim": len(vector),
        }
    finally:
        client.close()
