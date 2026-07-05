"""Configuration for Advanced RAG System."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

DOCUMENTS_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)

# ── LLM Provider ──────────────────────────────────────────────
# Options: "openrouter" (online) or "ollama" (local/offline)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")

# OpenRouter (online)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "mimollm/mimo-v2.5-free")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")

# Ollama (local, offline)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

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

# ── Hybrid Search (Dense + Sparse) ────────────────────────────
HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.5"))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.5"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "20"))

# ── Reranker (Cross-Encoder) ──────────────────────────────────
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
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
