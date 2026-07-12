"""Advanced RAG chain: query rewriting, hybrid retrieval, reranking, strict citation."""

import os
import re
import hashlib
import logging
import threading
from time import time
from typing import Dict, Any, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from config import (
    OPENROUTER_API_KEY,
    LLM_MODEL,
    LLM_BASE_URL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    NINEROUTER_API_KEY,
    NINEROUTER_BASE_URL,
    NINEROUTER_MODEL,
    LLM_PROVIDER,
    MIN_CITATION_COUNT,
    QUERY_REWRITING_ENABLED,
    TOP_K_RERANKED,
    LLM_TIMEOUT,
    CACHE_TTL,
)
from retrieval import HybridRetriever, Reranker

logger = logging.getLogger(__name__)




# ── Response Cache ──────────────────────────────────────────────

class ResponseCache:
    """Simple LRU cache with TTL for RAG responses."""

    def __init__(self, maxsize: int = 128, ttl: int = CACHE_TTL):
        self._cache: Dict[str, tuple] = {}
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = threading.Lock()

    def _key(self, question: str) -> str:
        return hashlib.sha256(question.strip().lower().encode()).hexdigest()

    def get(self, question: str) -> Optional[Dict[str, Any]]:
        key = self._key(question)
        with self._lock:
            entry = self._cache.get(key)
            if entry:
                result, ts = entry
                if time() - ts < self._ttl:
                    return result
                del self._cache[key]
        return None

    def set(self, question: str, result: Dict[str, Any]):
        key = self._key(question)
        with self._lock:
            if len(self._cache) >= self._maxsize:
                oldest = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest]
            self._cache[key] = (result, time())

    def clear(self):
        with self._lock:
            self._cache.clear()


cache = ResponseCache()


# ── Query Rewriter ──────────────────────────────────────────────

class QueryRewriter:
    """Rewrites/expands user queries for more accurate retrieval."""

    PROMPT = """Anda adalah asisten yang tugasnya memperbaiki pertanyaan user agar lebih jelas dan spesifik untuk pencarian dokumen.

Rules:
- Perbaiki tata bahasa dan ejaan
- Perluas singkatan atau akronim jika memungkinkan
- Ubah pertanyaan ambigu menjadi pertanyaan yang lebih spesifik
- Pertahankan intent asli
- Jawab HANYA dengan pertanyaan yang sudah diperbaiki, tanpa penjelasan tambahan
- Jika pertanyaan sudah jelas, kembalikan apa adanya

Pertanyaan asli: {question}
Pertanyaan yang diperbaiki:"""

    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, question: str) -> str:
        if not QUERY_REWRITING_ENABLED:
            return question
        try:
            prompt = ChatPromptTemplate.from_template(self.PROMPT)
            chain = prompt | self.llm | StrOutputParser()
            rewritten = chain.invoke({"question": question})
            return rewritten.strip()
        except Exception as e:
            logger.warning(f"Query rewriting failed (using original): {e}")
            return question


# ── Citation Validator ──────────────────────────────────────────

class CitationValidator:
    """Validates AI responses include proper document citations."""

    CITATION_PATTERNS = [
        r"\[Sumber:\s*.+,\s*Halaman\s*\d+\]",
        r"\[Source:\s*.+,\s*Page\s*\d+\]",
        r"Sumber:\s*.+\. Halaman\s*\d+",
        r"Berdasarkan dokumen\s*.+",
        r"\[Dokumen:\s*.+\]",
    ]

    UNCERTAINTY_PHRASES = [
        "saya tidak tahu",
        "informasi ini tidak tersedia",
        "tidak ada dalam dokumen",
        "tidak disebutkan",
        "tidak ditemukan",
        "tidak dapat menemukan",
    ]

    def validate(self, response: str, source_docs: List[Document]) -> Dict[str, Any]:
        has_citation = self._check_citation_exists(response)
        cited_files = self._extract_cited_files(response)
        all_source_files = [doc.metadata.get("filename", "unknown") for doc in source_docs]

        missing_sources = (
            [f for f in set(all_source_files) if f not in cited_files]
            if has_citation
            else all_source_files
        )

        is_uncertain = self._check_uncertainty(response)

        return {
            "has_citation": has_citation,
            "cited_files": list(cited_files),
            "missing_sources": missing_sources,
            "citation_count": len(cited_files),
            "is_uncertain": is_uncertain,
            "valid": (
                has_citation
                and len(cited_files) >= MIN_CITATION_COUNT
                and not (is_uncertain and has_citation)
            ),
        }

    def _check_citation_exists(self, text: str) -> bool:
        return any(re.search(p, text) for p in self.CITATION_PATTERNS)

    def _check_uncertainty(self, text: str) -> bool:
        lower = text.lower()
        return any(phrase in lower for phrase in self.UNCERTAINTY_PHRASES)

    def _extract_cited_files(self, text: str) -> set:
        cited = set()
        patterns = [
            r"\[Sumber:\s*([^,\]]+)",
            r"\[Source:\s*([^,\]]+)",
            r"\[Dokumen:\s*([^,\]]+)",
            r"Sumber:\s*([^.]+)\.",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            cited.update(m.strip() for m in matches)
        return cited


# ── LLM Factory ─────────────────────────────────────────────────

def _create_llm():
    """Create LLM with timeout to prevent hanging."""
    kwargs = dict(
        temperature=0.1,
        max_tokens=2048,
        request_timeout=LLM_TIMEOUT,
    )
    if LLM_PROVIDER == "ollama":
        return ChatOpenAI(
            model=OLLAMA_MODEL,
            base_url=f"{OLLAMA_BASE_URL}/v1",
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            **kwargs,
        )
    elif LLM_PROVIDER == "9router":
        return ChatOpenAI(
            model=NINEROUTER_MODEL,
            base_url=NINEROUTER_BASE_URL,
            api_key=NINEROUTER_API_KEY,
            **kwargs,
        )
    # Default: OpenRouter
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        **kwargs,
    )


# ── RAG Chain ───────────────────────────────────────────────────

class RAGChain:
    """Full Advanced RAG pipeline: rewrite -> hybrid search -> rerank -> generate -> validate."""

    SYSTEM_PROMPT = """Anda adalah asisten AI yang menjawab pertanyaan HANYA berdasarkan dokumen yang diberikan.

ATURAN:
1. Jawab HANYA berdasarkan informasi dalam dokumen yang disediakan.
2. Jika informasi tidak ada di dokumen, katakan "Informasi ini tidak tersedia dalam dokumen yang diberikan."
3. Sertakan sitasi/sumber untuk SETIAP klaim informasi menggunakan format:
   [Sumber: nama_file.pdf, Halaman X]
4. Jika informasi berasal dari beberapa dokumen, cantumkan SEMUA sumbernya.
5. Jangan mengarang informasi di luar dokumen. Jika ragu, akui ketidaktahuan.

Format jawaban:
- Mulai dengan jawaban singkat
- Sertakan sitasi di setiap kalimat yang merujuk dokumen
- Akhiri dengan ringkasan sumber yang digunakan"""

    USER_PROMPT = """Konteks dokumen:
{context}

Pertanyaan: {question}

Jawab dengan selalu menyertakan sumber/sitasi dokumen."""

    def __init__(self, hybrid_retriever: HybridRetriever, reranker: Reranker):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.llm = _create_llm()
        self.rewriter = QueryRewriter(self.llm)
        self.validator = CitationValidator()
        self.chain = self._build_chain()

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", self.USER_PROMPT),
        ])
        return (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs: List[Document]) -> str:
        formatted = []
        for doc in docs:
            source = doc.metadata.get("filename", "unknown")
            page = doc.metadata.get("page", 0)
            page_label = page + 1 if isinstance(page, int) else page
            formatted.append(
                f"[Dokumen: {source}, Halaman {page_label}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted)

    def query(self, question: str) -> Dict[str, Any]:
        """Full RAG pipeline with caching and timeout protection."""
        MAX_ATTEMPTS = 2
        PIPELINE_TIMEOUT = 55

        cached = cache.get(question)
        if cached:
            return cached

        rewritten = self.rewriter.rewrite(question)

        last_response = None
        last_validation = None

        for attempt in range(MAX_ATTEMPTS):
            try:
                raw_docs = self.hybrid_retriever.retrieve(rewritten, k=TOP_K_RERANKED * 4)
                reranked_docs = self.reranker.rerank(rewritten, raw_docs, top_k=TOP_K_RERANKED)

                if not reranked_docs:
                    result = {
                        "answer": "Tidak ada dokumen yang relevan ditemukan untuk pertanyaan ini.",
                        "rewritten_query": rewritten,
                        "sources": [],
                        "validation": {"valid": True, "citation_count": 0, "has_citation": False, "cited_files": [], "missing_sources": [], "is_uncertain": False},
                        "attempts": 1,
                    }
                    cache.set(question, result)
                    return result

                context = self._format_docs(reranked_docs)

                current_question = (
                    f"Jawaban Anda sebelumnya kurang lengkap dalam menyertakan sitasi. "
                    f"PASTIKAN setiap fakta memiliki sitasi [Sumber: nama_file.pdf, Halaman X].\n"
                    f"Pertanyaan: {rewritten}"
                    if attempt > 0
                    else rewritten
                )

                raw_response = self.chain.invoke({
                    "context": context,
                    "question": current_question,
                })

                if not raw_response or not raw_response.strip():
                    continue

                last_response = raw_response
                last_validation = self.validator.validate(raw_response, reranked_docs)

                if last_validation["valid"]:
                    result = {
                        "answer": raw_response,
                        "rewritten_query": rewritten,
                        "sources": reranked_docs,
                        "validation": last_validation,
                        "attempts": attempt + 1,
                    }
                    cache.set(question, result)
                    return result

            except Exception as e:
                logger.error(f"Pipeline attempt {attempt + 1} failed: {e}")
                if attempt == MAX_ATTEMPTS - 1:
                    raise

        result = {
            "answer": last_response or "Maaf, tidak dapat menghasilkan jawaban. Silakan coba lagi.",
            "rewritten_query": rewritten,
            "sources": reranked_docs if reranked_docs else [],
            "validation": last_validation or {"valid": False, "citation_count": 0, "has_citation": False, "cited_files": [], "missing_sources": [], "is_uncertain": False},
            "attempts": MAX_ATTEMPTS,
        }
        cache.set(question, result)
        return result
