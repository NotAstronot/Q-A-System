"""Configuration for Internal Q&A System."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# Ensure directories exist
DOCUMENTS_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)

# LLM Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    import warnings
    warnings.warn("OPENROUTER_API_KEY is not set. LLM queries will fail.")
LLM_MODEL = os.getenv("LLM_MODEL", "mimollm/mimo-v2.5-free")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")

# ChromaDB Settings
CHROMA_COLLECTION_NAME = "internal_docs"
CHROMA_PERSIST_DIR = str(CHROMA_DB_DIR)

# RAG Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))

# Embedding Model (using free alternative)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Citation Settings
MIN_CITATION_COUNT = int(os.getenv("MIN_CITATION_COUNT", "1"))
CITATION_FORMAT = "[Sumber: {filename}, Halaman {page}]"

# API Settings
API_KEY = os.getenv("API_KEY", "")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
