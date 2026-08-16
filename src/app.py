import threading
from contextlib import asynccontextmanager

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from admin_api import router as admin_router
from config import ALL_COLLECTIONS, DEFAULT_TOP_K
from ingestion import ingest_existing_files, start_watcher
from query import search


@asynccontextmanager
async def lifespan(app: FastAPI):
    ingest_existing_files()
    threading.Thread(target=start_watcher, daemon=True).start()
    yield


app = FastAPI(title="RAG Pipeline API", lifespan=lifespan)
app.include_router(admin_router)
app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")


class RetrieveRequest(BaseModel):
    question: str
    # Required until a routing agent picks the collection automatically.
    collection: str  # one of config.ALL_COLLECTIONS, e.g. "AURORA_PRODUCT"
    search_type: str = "hybrid"  # "keyword", "vector", or "hybrid"
    k: int = DEFAULT_TOP_K


class ContextChunk(BaseModel):
    source_file: Optional[str] = None
    doc_type: Optional[str] = None
    page_number: Optional[int] = None
    content_type: Optional[str] = None
    table_name: Optional[str] = None
    section_heading: Optional[str] = None
    product_name: Optional[str] = None
    text: str


class RetrieveResponse(BaseModel):
    question: str
    search_type: str
    context: List[ContextChunk]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/")


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest):
    """Search primitive consumed by the rag-retrieval service: returns the
    matching chunks (vector/keyword/hybrid) with no LLM call involved.
    """
    if request.collection not in ALL_COLLECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"collection must be one of {ALL_COLLECTIONS}, got {request.collection!r}")
    context = search(request.question, collection=request.collection,
                     search_type=request.search_type, k=request.k)
    return RetrieveResponse(
        question=request.question, search_type=request.search_type, context=context)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
