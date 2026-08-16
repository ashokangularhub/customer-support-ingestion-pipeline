import threading
from contextlib import asynccontextmanager

from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from admin_api import router as admin_router
from ingestion import ingest_existing_files, start_watcher
from query import ask_question


@asynccontextmanager
async def lifespan(app: FastAPI):
    ingest_existing_files()
    threading.Thread(target=start_watcher, daemon=True).start()
    yield


app = FastAPI(title="RAG Pipeline API", lifespan=lifespan)
app.include_router(admin_router)
app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")


class QueryRequest(BaseModel):
    question: str
    search_type: str = "hybrid"  # "keyword", "vector", or "hybrid"


class ContextChunk(BaseModel):
    source_file: Optional[str] = None
    doc_type: Optional[str] = None
    page_number: Optional[int] = None
    content_type: Optional[str] = None
    table_name: Optional[str] = None
    section_heading: Optional[str] = None
    product_name: Optional[str] = None
    text: str


class QueryResponse(BaseModel):
    system_prompt: str
    context: List[ContextChunk]
    question: str
    search_type: str
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    result = ask_question(request.question, search_type=request.search_type)
    return QueryResponse(**result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
