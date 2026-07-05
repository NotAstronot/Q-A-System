"""Document ingestion: parent-child chunking, BM25 indexing, table parsing."""

import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import (
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    TABLE_PARSING_ENABLED,
)


def _table_to_markdown(table_data: list) -> str:
    """Convert pdfplumber table data to markdown table."""
    if not table_data:
        return ""
    rows = []
    for row in table_data:
        cells = [str(cell or "").strip() for cell in row]
        rows.append("| " + " | ".join(cells) + " |")
    if rows:
        header = rows[0]
        num_cols = len(header.split("|")) - 2
        sep = "| " + " | ".join(["---"] * max(num_cols, 1)) + " |"
        rows.insert(1, sep)
    return "\n".join(rows)


def _extract_tables_from_page(page) -> str:
    """Extract tables from a pdfplumber page as markdown."""
    tables = page.extract_tables()
    if not tables:
        return ""
    sections = []
    for table in tables:
        md = _table_to_markdown(table)
        if md:
            sections.append(md)
    return "\n\n".join(sections)


class DocumentIngestor:
    """Handles PDF loading, parent-child chunking, BM25 index, and ChromaDB storage."""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PARENT_CHUNK_SIZE,
            chunk_overlap=PARENT_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE,
            chunk_overlap=CHILD_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.vectorstore: Optional[Chroma] = None
        self.bm25_corpus: List[str] = []
        self.bm25_metadata: List[dict] = []
        self.bm25_index_path = Path(CHROMA_PERSIST_DIR) / "bm25_corpus.pkl"
        self.bm25_meta_path = Path(CHROMA_PERSIST_DIR) / "bm25_metadata.pkl"

    # ── PDF Loading ──────────────────────────────────────────────

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load PDF text + optional table extraction."""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if TABLE_PARSING_ENABLED:
            try:
                documents = self._inject_table_content(pdf_path, documents)
            except Exception:
                pass

        filename = Path(pdf_path).name
        for doc in documents:
            doc.metadata["source_type"] = "pdf"
            doc.metadata["filename"] = filename
        return documents

    def _inject_table_content(self, pdf_path: str, documents: List[Document]) -> List[Document]:
        """Use pdfplumber to extract tables and inject as markdown into page text."""
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for doc in documents:
                page_num = doc.metadata.get("page", 0)
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]
                    table_md = _extract_tables_from_page(page)
                    if table_md:
                        doc.page_content += "\n\n[TABEL]\n" + table_md + "\n[/TABEL]"
        return documents

    # ── Parent-Child Chunking ────────────────────────────────────

    def create_parent_child_chunks(self, documents: List[Document]) -> Tuple[List[Document], List[str]]:
        """Split into parent chunks, then child chunks. Children store parent content in metadata."""
        parent_docs = self.parent_splitter.split_documents(documents)
        for i, doc in enumerate(parent_docs):
            doc.metadata["parent_chunk_index"] = i

        child_chunks: List[Document] = []
        parent_contents: List[str] = []

        for parent in parent_docs:
            children = self.child_splitter.split_documents([parent])
            parent_text = parent.page_content
            parent_contents.append(parent_text)
            for child in children:
                child.metadata["parent_content"] = parent_text
                child.metadata["parent_chunk_index"] = parent.metadata["parent_chunk_index"]
                child.metadata["parent_length"] = len(parent_text)
            child_chunks.extend(children)

        return child_chunks, parent_contents

    # ── BM25 Index ───────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer."""
        return re.sub(r"[^\w\s]", " ", text.lower()).split()

    def _build_bm25_corpus(self, child_chunks: List[Document]) -> Tuple[List[str], List[dict]]:
        """Build text corpus + metadata for BM25 from child chunks."""
        texts = [chunk.page_content for chunk in child_chunks]
        metadata = [dict(chunk.metadata) for chunk in child_chunks]
        return texts, metadata

    def save_bm25_corpus(self, corpus: List[str], metadata: List[dict]):
        """Persist BM25 corpus and metadata to disk."""
        with open(self.bm25_index_path, "wb") as f:
            pickle.dump(corpus, f)
        with open(self.bm25_meta_path, "wb") as f:
            pickle.dump(metadata, f)

    def load_bm25_corpus(self) -> Tuple[List[str], List[dict]]:
        """Load BM25 corpus and metadata from disk."""
        corpus, metadata = [], []
        if self.bm25_index_path.exists():
            with open(self.bm25_index_path, "rb") as f:
                corpus = pickle.load(f)
        if self.bm25_meta_path.exists():
            with open(self.bm25_meta_path, "rb") as f:
                metadata = pickle.load(f)
        return corpus, metadata

    # ── Main Ingestion Pipeline ──────────────────────────────────

    def ingest_pdf(self, pdf_path: str) -> int:
        """Full pipeline: load -> table parse -> parent-child chunk -> store in ChromaDB + BM25."""
        filename = Path(pdf_path).name

        if self.vectorstore is not None:
            existing = self.vectorstore.get(where={"filename": filename})
            if existing and existing.get("ids"):
                self.vectorstore.delete(ids=existing["ids"])

        documents = self.load_pdf(pdf_path)
        child_chunks, parent_texts = self.create_parent_child_chunks(documents)

        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=child_chunks,
                embedding=self.embeddings,
                collection_name=CHROMA_COLLECTION_NAME,
                persist_directory=CHROMA_PERSIST_DIR,
            )
        else:
            self.vectorstore.add_documents(child_chunks)

        corpus, metadata = self._build_bm25_corpus(child_chunks)
        self.bm25_corpus = corpus
        self.bm25_metadata = metadata
        self.save_bm25_corpus(corpus, metadata)

        return len(child_chunks)

    def ingest_directory(self, dir_path: str) -> dict:
        """Ingest all PDFs in a directory."""
        results = {}
        pdf_files = list(Path(dir_path).glob("*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {dir_path}")
            return results

        for pdf_file in pdf_files:
            try:
                chunk_count = self.ingest_pdf(str(pdf_file))
                results[pdf_file.name] = chunk_count
                print(f"  [OK] {pdf_file.name}: {chunk_count} child chunks")
            except Exception as e:
                results[pdf_file.name] = -1
                print(f"  [ERR] {pdf_file.name}: {e}")

        return results

    # ── Retriever Access ────────────────────────────────────────

    def rebuild_bm25_from_store(self) -> Tuple[List[str], List[dict]]:
        """Rebuild BM25 corpus from ChromaDB on startup."""
        corpus, metadata = self.load_bm25_corpus()
        if not corpus:
            if self.vectorstore is None:
                self.vectorstore = Chroma(
                    collection_name=CHROMA_COLLECTION_NAME,
                    embedding_function=self.embeddings,
                    persist_directory=CHROMA_PERSIST_DIR,
                )
            all_docs = self.vectorstore.get()
            if all_docs and all_docs.get("documents"):
                docs_text = all_docs["documents"]
                docs_meta = all_docs.get("metadatas", [{}] * len(docs_text))
                corpus = docs_text
                metadata = docs_meta
                self.save_bm25_corpus(corpus, metadata)
        self.bm25_corpus = corpus
        self.bm25_metadata = metadata
        return corpus, metadata

    def get_vectorstore(self) -> Chroma:
        """Return Chroma vector store instance."""
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=CHROMA_PERSIST_DIR,
            )
        return self.vectorstore

    def get_bm25_data(self) -> Tuple[List[str], List[dict]]:
        """Return BM25 corpus + metadata, rebuilding from disk if needed."""
        if not self.bm25_corpus:
            self.rebuild_bm25_from_store()
        return self.bm25_corpus, self.bm25_metadata
