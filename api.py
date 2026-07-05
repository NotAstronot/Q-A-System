"""FastAPI server for Advanced RAG System."""

import re
import logging
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "2.0.0"

app = FastAPI(
    title="Internal Q&A System - Advanced RAG",
    description="Advanced RAG system with hybrid search, reranking, query rewriting, parent-child chunking",
    version=VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

ingestor = DocumentIngestor()
rag_chain: Optional[RAGChain] = None


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


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


def get_api_key(api_key: str = Security(api_key_header)):
    if not API_KEY:
        return
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def get_rag_chain() -> RAGChain:
    global rag_chain
    if rag_chain is None:
        vectorstore = ingestor.get_vectorstore()
        corpus, metadata = ingestor.get_bm25_data()
        bm25 = BM25Retriever(corpus, metadata)
        hybrid = HybridRetriever(vectorstore, bm25)
        reranker = Reranker(RERANKER_MODEL)
        rag_chain = RAGChain(hybrid, reranker)
    return rag_chain


# ── Endpoints ────────────────────────────────────────────────


@app.get("/api/health", response_model=HealthResponse)
def health_check():
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
def query_documents(req: QueryRequest, _key: str = Depends(get_api_key)):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    chain = get_rag_chain()
    result = chain.query(req.question)

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
async def ingest_document(file: UploadFile = File(...), _key: str = Depends(get_api_key)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are allowed")

    safe_name = re.sub(r'[^\w\-. ]', '', file.filename)
    safe_name = Path(safe_name).name
    if not safe_name:
        raise HTTPException(status_code=422, detail="Invalid filename")

    save_path = DOCUMENTS_DIR / safe_name
    if not save_path.resolve().is_relative_to(DOCUMENTS_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename path")

    content = await file.read()

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB")

    with open(save_path, "wb") as f:
        f.write(content)

    try:
        chunk_count = ingestor.ingest_pdf(str(save_path))
        global rag_chain
        rag_chain = None
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
        logger.error(f"Ingestion failed for {safe_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


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
    for pdf_file in DOCUMENTS_DIR.glob("*.pdf"):
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
        file_path = frontend_dir / "dist" / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(frontend_dir / "dist" / "index.html"))
else:
    logger.warning("Frontend not built. Run 'cd frontend && npm run build' to serve the UI.")
