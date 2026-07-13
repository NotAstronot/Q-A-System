"""Configuration for Advanced RAG System."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
LOGS_DIR = BASE_DIR / "logs"

DOCUMENTS_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ── LLM Provider ──────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")

# 9router Proxy
NINEROUTER_API_KEY = os.getenv("NINEROUTER_API_KEY", "")
NINEROUTER_BASE_URL = os.getenv("NINEROUTER_BASE_URL", "https://api.9router.com/v1")
NINEROUTER_MODEL = os.getenv("NINEROUTER_MODEL", "gpt-4-turbo")

# OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "mimollm/mimo-v2.5-free")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

# ── Vector Database ───────────────────────────────────────────
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chromadb")  # or "qdrant"
QDRANT_PATH = os.getenv("QDRANT_PATH", str(BASE_DIR / "qdrant_db"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "internal_docs")

# ── ChromaDB ──────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "internal_docs"
CHROMA_PERSIST_DIR = str(CHROMA_DB_DIR)

# ── Embedding Model ───────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ── Parent-Child Chunking ─────────────────────────────────────
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "1000"))
PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", "200"))
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "250"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "50"))

# ── Hybrid Search ────────────────────────────────────────────
HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.5"))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.5"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "20"))

# ── Reranker ──────────────────────────────────────────────────
RERANKING_ENABLED = os.getenv("RERANKING_ENABLED", "true").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-4-v2")
TOP_K_RERANKED = int(os.getenv("TOP_K_RERANKED", "5"))

# ── Query Rewriting ───────────────────────────────────────────
QUERY_REWRITING_ENABLED = os.getenv("QUERY_REWRITING_ENABLED", "true").lower() == "true"

# ── Table Parsing ─────────────────────────────────────────────
TABLE_PARSING_ENABLED = os.getenv("TABLE_PARSING_ENABLED", "true").lower() == "true"

# ── Citation ──────────────────────────────────────────────────
MIN_CITATION_COUNT = int(os.getenv("MIN_CITATION_COUNT", "1"))
CITATION_FORMAT = "[Sumber: {filename}, Halaman {page}]"

# ── API ───────────────────────────────────────────────────────
API_KEY = os.getenv("API_KEY", "")
CORS_ORIGINS_STR = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
CORS_ORIGINS = [o.strip() for o in CORS_ORIGINS_STR.split(",") if o.strip()]

# ── Performance ────────────────────────────────────────────────
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))

# ── Performance Optimizations ──────────────────────────────────
# Redis Caching
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() == "true"
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "3600"))

# GPU Acceleration
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
USE_FP16 = os.getenv("USE_FP16", "false").lower() == "true"

# Response Streaming
ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "false").lower() == "true"

# ── Rate Limiting ─────────────────────────────────────────────
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "60"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
