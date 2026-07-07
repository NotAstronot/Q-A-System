"""Hybrid retriever (dense + sparse) and Cross-Encoder reranker."""

import re
import pickle
import logging
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import (
    RERANKER_MODEL,
    TOP_K_RETRIEVAL,
    TOP_K_RERANKED,
    DENSE_WEIGHT,
    SPARSE_WEIGHT,
    RERANKING_ENABLED,
    HYBRID_SEARCH_ENABLED,
    CHROMA_PERSIST_DIR,
)

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


class BM25Retriever:
    """BM25 sparse retriever using rank_bm25."""

    def __init__(self, corpus: List[str], doc_metadata: Optional[List[dict]] = None):
        self.corpus = corpus
        self.doc_metadata = doc_metadata or []
        tokenized = [_tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def retrieve(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Tuple[int, float]]:
        if not self.bm25 or not self.corpus:
            return []
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


class HybridRetriever:
    """Combine dense (ChromaDB) and sparse (BM25) retrieval with RRF fusion."""

    def __init__(self, vectorstore: Chroma, bm25_retriever: BM25Retriever):
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever

    def _rrf_score(self, dense_rank: Optional[int], sparse_rank: Optional[int], k: int = 60) -> float:
        score = 0.0
        if dense_rank is not None:
            score += DENSE_WEIGHT / (k + dense_rank + 1)
        if sparse_rank is not None:
            score += SPARSE_WEIGHT / (k + sparse_rank + 1)
        return score

    def retrieve(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Document]:
        """Hybrid search: dense + sparse with RRF fusion."""
        corpus = self.bm25_retriever.corpus
        doc_meta = self.bm25_retriever.doc_metadata

        dense_pairs = self.vectorstore.similarity_search_with_score(query, k=k)
        dense_docs = [doc for doc, _ in dense_pairs]

        sparse_pairs = self.bm25_retriever.retrieve(query, k=k)
        sparse_indices = {idx: rank for rank, (idx, _) in enumerate(sparse_pairs)}

        scored_docs: List[Tuple[Document, float]] = []
        seen_indices: set = set()

        for rank, doc in enumerate(dense_docs):
            dense_rank = rank
            sparse_rank = sparse_indices.get(rank)
            rrf = self._rrf_score(dense_rank, sparse_rank)
            scored_docs.append((doc, rrf))
            seen_indices.add(rank)

        for corpus_idx, _ in sparse_pairs:
            if corpus_idx not in seen_indices and corpus_idx < len(corpus):
                text = corpus[corpus_idx]
                meta = doc_meta[corpus_idx] if corpus_idx < len(doc_meta) else {}
                doc = Document(page_content=text, metadata=meta)
                sparse_rank = sparse_indices.get(corpus_idx)
                rrf = self._rrf_score(None, sparse_rank)
                scored_docs.append((doc, rrf))
                seen_indices.add(corpus_idx)

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        expanded = []
        for doc, _ in scored_docs[:k]:
            parent_content = doc.metadata.get("parent_content", "")
            if parent_content:
                expanded.append(Document(
                    page_content=parent_content,
                    metadata=doc.metadata,
                ))
            else:
                expanded.append(doc)

        return expanded


class Reranker:
    """Cross-Encoder reranker with lazy loading and GPU support."""

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model = None
        self.model_name = model_name
        self._lock = threading.Lock()
        self._loaded = False
        self._load_path = Path(CHROMA_PERSIST_DIR) / "reranker_loaded.flag"

    def preload(self):
        """Preload model in background thread to speed up first query."""
        if self._loaded or not RERANKING_ENABLED:
            return
        t = threading.Thread(target=self._lazy_load, daemon=True)
        t.start()

    def _lazy_load(self):
        if self.model is not None:
            return
        with self._lock:
            if self.model is not None:
                return
            if not RERANKING_ENABLED:
                self._loaded = True
                return
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading reranker model: {self.model_name}")
                self.model = CrossEncoder(
                    self.model_name,
                    device="cpu",
                )
                self._loaded = True
                logger.info("Reranker model loaded")
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}")
                self._loaded = True

    def rerank(self, query: str, documents: List[Document], top_k: int = TOP_K_RERANKED) -> List[Document]:
        if not documents or not RERANKING_ENABLED:
            return documents[:top_k]

        self._lazy_load()
        if self.model is None:
            return documents[:top_k]

        try:
            pairs = [[query, doc.page_content] for doc in documents]
            scores = self.model.predict(pairs, show_progress_bar=False)
            scored = list(zip(documents, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored[:top_k]]
        except Exception as e:
            logger.warning(f"Reranker failed, returning unranked: {e}")
            return documents[:top_k]
