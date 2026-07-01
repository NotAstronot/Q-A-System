"""Document ingestion module using PyPDF and ChromaDB."""

from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
)


class DocumentIngestor:
    """Handles PDF loading, chunking, and ChromaDB ingestion."""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.vectorstore = None

    def load_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Load PDF and extract text with metadata."""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source_type"] = "pdf"
            doc.metadata["filename"] = Path(pdf_path).name
        return documents

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into smaller chunks for embedding."""
        chunks = self.text_splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        return chunks

    def ingest_pdf(self, pdf_path: str) -> int:
        """Full pipeline: load PDF -> chunk -> store in ChromaDB."""
        documents = self.load_pdf(pdf_path)
        chunks = self.chunk_documents(documents)

        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=CHROMA_COLLECTION_NAME,
                persist_directory=CHROMA_PERSIST_DIR,
            )
        else:
            self.vectorstore.add_documents(chunks)

        return len(chunks)

    def ingest_directory(self, dir_path: str) -> Dict[str, int]:
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
                print(f"  [OK] {pdf_file.name}: {chunk_count} chunks")
            except Exception as e:
                results[pdf_file.name] = -1
                print(f"  [ERR] {pdf_file.name}: {e}")

        return results

    def get_retriever(self):
        """Return a retriever from the vector store."""
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=CHROMA_PERSIST_DIR,
            )
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_RESULTS},
        )
