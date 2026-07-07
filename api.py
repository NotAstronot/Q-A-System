"""FastAPI server for Advanced RAG System."""

import re
import hmac
import logging
from time import time
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from config import (
    DOCUMENTS_DIR,
    CORS_ORIGINS,
    API_KEY,
    LLM_PROVIDER,
    RERANKER_MODEL,
    RERANKING_ENABLED,
    QUERY_REWRITING_ENABLED,
    HYBRID_SEARCH_ENABLED,
    TABLE_PARSING_ENABLED,
    OLLAMA_MODEL,
    LLM_MODEL,
)
from ingestion import DocumentIngestor
from retrieval import HybridRetriever, BM25Retriever, Reranker
from chain import RAGChain
from security import (
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    sanitize_filename,
    sanitize_question,
    validate_file_content,
    check_brute_force,
    reset_brute_force,
    audit,
    rate_limiter,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "2.0.0"

app = FastAPI(
    title="Internal Q&A System - Advanced RAG",
    description="Advanced RAG system with hybrid search, reranking, query rewriting, parent-child chunking",
    version=VERSION,
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
    expose_headers=["X-RateLimit-Remaining", "X-RateLimit-Reset"],
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

ingestor = DocumentIngestor()
rag_chain: Optional[RAGChain] = None


class QueryRequest(BaseModel):
    question: str = Field(..., max_length=4096, min_length=1)
    top_k: Optional[int] = Field(None, ge=1, le=50)


class SourceItem(BaseModel):
    filename: str
    page: int
    content: str
    parent_length: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    rewritten_query: Optional[str] = None
    sources: list
    validation: dict
    attempts: int


class IngestResponse(BaseModel):
    filename: str
    chunks_created: int
    status: str


class StatsResponse(BaseModel):
    collection_name: str
    total_chunks: int
    documents_dir: str
    bm25_trained: bool
    provider: str


class DocumentItem(BaseModel):
    filename: str
    size_kb: float


class DocumentsResponse(BaseModel):
    documents: list[DocumentItem]


class HealthResponse(BaseModel):
    status: str
    version: str
    provider: str
    features: dict


class FeaturesResponse(BaseModel):
    hybrid_search: bool
    reranking: bool
    reranker_model: str
    query_rewriting: bool
    table_parsing: bool
    provider: str
    model: str
    parent_child_chunking: bool


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    audit.log("INTERNAL_ERROR", detail=str(exc), severity="ERROR")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


def get_api_key(api_key: str = Security(api_key_header), request: Request = None):
    if not API_KEY:
        return
    ip = request.client.host if request and request.client else "unknown"
    if check_brute_force(ip):
        audit.log("BRUTE_FORCE_DETECTED", detail="Multiple failed auth attempts", ip=ip, severity="ERROR")
        raise HTTPException(status_code=429, detail="Too many authentication failures")
    if not api_key:
        audit.log("AUTH_FAILED", detail="Missing API key", ip=ip, severity="WARN")
        raise HTTPException(status_code=401, detail="Missing API key")
    if not hmac.compare_digest(api_key, API_KEY):
        audit.log("AUTH_FAILED", detail="Invalid API key", ip=ip, severity="WARN")
        raise HTTPException(status_code=401, detail="Invalid API key")
    reset_brute_force(ip)
    return api_key


def get_rag_chain() -> RAGChain:
    global rag_chain
    if rag_chain is None:
        t0 = time()
        logger.info("Initializing RAG chain...")
        vectorstore = ingestor.get_vectorstore()
        corpus, metadata = ingestor.get_bm25_data()
        bm25 = BM25Retriever(corpus, metadata)
        hybrid = HybridRetriever(vectorstore, bm25)
        reranker = Reranker(RERANKER_MODEL)
        reranker.preload()
        rag_chain = RAGChain(hybrid, reranker)
        logger.info(f"RAG chain initialized in {time() - t0:.1f}s")
    return rag_chain


# ── Endpoints ────────────────────────────────────────────────


@app.get("/api/health", response_model=HealthResponse)
def health_check(request: Request):
    ip = request.client.host if request.client else "unknown"
    audit.log("HEALTH_CHECK", ip=ip)
    return HealthResponse(
        status="ok",
        version=VERSION,
        provider=LLM_PROVIDER,
        features={
            "hybrid_search": HYBRID_SEARCH_ENABLED,
            "reranking": RERANKING_ENABLED,
            "query_rewriting": QUERY_REWRITING_ENABLED,
            "table_parsing": TABLE_PARSING_ENABLED,
        },
    )


@app.get("/api/config")
def get_config():
    return {
        "api_key": API_KEY,
        "has_api_key": bool(API_KEY),
    }


@app.get("/api/features", response_model=FeaturesResponse)
def get_features():
    model = OLLAMA_MODEL if LLM_PROVIDER == "ollama" else LLM_MODEL
    return FeaturesResponse(
        hybrid_search=HYBRID_SEARCH_ENABLED,
        reranking=RERANKING_ENABLED,
        reranker_model=RERANKER_MODEL if RERANKING_ENABLED else "disabled",
        query_rewriting=QUERY_REWRITING_ENABLED,
        table_parsing=TABLE_PARSING_ENABLED,
        provider=LLM_PROVIDER,
        model=model,
        parent_child_chunking=True,
    )


@app.post("/api/query", response_model=QueryResponse)
def query_documents(req: QueryRequest, _key: str = Depends(get_api_key), request: Request = None):
    t_start = time()
    question = sanitize_question(req.question)
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    ip = request.client.host if request and request.client else "unknown"
    audit.log("QUERY", detail=f"question_preview={question[:100]}", ip=ip)

    chain = get_rag_chain()
    try:
        result = chain.query(question)
    except Exception as e:
        audit.log("QUERY_FAILED", detail=str(e), ip=ip, severity="ERROR")
        raise HTTPException(status_code=500, detail="Query processing failed")

    elapsed = time() - t_start
    logger.info(f"Query completed in {elapsed:.1f}s | attempts={result.get('attempts', '?')} | valid={result.get('validation', {}).get('valid')}")

    sources = []
    for doc in result["sources"]:
        sources.append({
            "filename": doc.metadata.get("filename", "unknown"),
            "page": doc.metadata.get("page", 0),
            "content": doc.page_content[:500],
            "parent_length": doc.metadata.get("parent_length"),
        })

    return QueryResponse(
        answer=result["answer"],
        rewritten_query=result.get("rewritten_query"),
        sources=sources,
        validation=result["validation"],
        attempts=result["attempts"],
    )


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...), _key: str = Depends(get_api_key), request: Request = None):
    ip = request.client.host if request and request.client else "unknown"

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        audit.log("INGEST_REJECTED", detail="Non-PDF file rejected", ip=ip, severity="WARN")
        raise HTTPException(status_code=422, detail="Only PDF files are allowed")

    safe_name = sanitize_filename(file.filename)
    if not safe_name:
        raise HTTPException(status_code=422, detail="Invalid filename")

    save_path = DOCUMENTS_DIR / safe_name
    if not save_path.resolve().is_relative_to(DOCUMENTS_DIR.resolve()):
        audit.log("INGEST_REJECTED", detail="Path traversal detected", ip=ip, severity="ERROR")
        raise HTTPException(status_code=400, detail="Invalid filename path")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB")

    if not validate_file_content(content):
        audit.log("INGEST_REJECTED", detail="Invalid PDF content (magic bytes)", ip=ip, severity="WARN")
        raise HTTPException(status_code=422, detail="File is not a valid PDF")

    temp_path = save_path.with_suffix(".tmp")
    try:
        with open(temp_path, "wb") as f:
            f.write(content)
        temp_path.rename(save_path)

        chunk_count = ingestor.ingest_pdf(str(save_path))
        global rag_chain
        rag_chain = None

        audit.log("INGEST_SUCCESS", detail=f"{safe_name}: {chunk_count} chunks", ip=ip)
        return IngestResponse(
            filename=safe_name,
            chunks_created=chunk_count,
            status="success",
        )
    except Exception as e:
        try:
            save_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        logger.error(f"Ingestion failed for {safe_name}: {e}")
        audit.log("INGEST_FAILED", detail=str(e), ip=ip, severity="ERROR")
        raise HTTPException(status_code=500, detail="Ingestion failed")


@app.get("/api/stats", response_model=StatsResponse)
def get_stats(_key: str = Depends(get_api_key)):
    vectorstore = ingestor.get_vectorstore()
    collection = vectorstore._collection
    corpus, _ = ingestor.get_bm25_data()
    return StatsResponse(
        collection_name=collection.name,
        total_chunks=collection.count(),
        documents_dir=str(DOCUMENTS_DIR),
        bm25_trained=len(corpus) > 0,
        provider=LLM_PROVIDER,
    )


@app.get("/api/documents", response_model=DocumentsResponse)
def list_documents(_key: str = Depends(get_api_key)):
    documents = []
    for pdf_file in sorted(DOCUMENTS_DIR.glob("*.pdf")):
        size_kb = pdf_file.stat().st_size / 1024
        documents.append(DocumentItem(
            filename=pdf_file.name,
            size_kb=round(size_kb, 2),
        ))
    return DocumentsResponse(documents=documents)


frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists() and (frontend_dir / "dist").exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dir / "dist" / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if not full_path:
            return FileResponse(str(frontend_dir / "dist" / "index.html"))
        resolved = (frontend_dir / "dist" / full_path).resolve()
        dist_resolved = (frontend_dir / "dist").resolve()
        if not str(resolved).startswith(str(dist_resolved)):
            raise HTTPException(status_code=404, detail="Not found")
        if resolved.is_file():
            return FileResponse(str(resolved))
        return FileResponse(str(frontend_dir / "dist" / "index.html"))
else:
    logger.warning("Frontend not built. Run 'cd frontend && npm run build' to serve the UI.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info",
    )
