"""RAG chain with citation validation using LangChain + Mimo V2.5."""

import re
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from config import (
    OPENROUTER_API_KEY,
    LLM_MODEL,
    LLM_BASE_URL,
    CITATION_FORMAT,
    MIN_CITATION_COUNT,
)


class CitationValidator:
    """Validates that AI responses include proper document citations."""

    CITATION_PATTERNS = [
        r"\[Sumber:\s*.+,\s*Halaman\s*\d+\]",
        r"\[Source:\s*.+,\s*Page\s*\d+\]",
        r"Sumber:\s*.+\. Halaman\s*\d+",
        r"Berdasarkan dokumen\s*.+",
    ]

    def validate(self, response: str, source_docs: List[Document]) -> Dict[str, Any]:
        """Validate citation presence and quality in response."""
        has_citation = self._check_citation_exists(response)
        cited_files = self._extract_cited_files(response)
        all_source_files = [doc.metadata.get("filename", "unknown") for doc in source_docs]

        missing_sources = [
            f for f in set(all_source_files)
            if f not in cited_files
        ] if has_citation else all_source_files

        return {
            "has_citation": has_citation,
            "cited_files": list(cited_files),
            "missing_sources": missing_sources,
            "citation_count": len(cited_files),
            "valid": has_citation and len(cited_files) >= MIN_CITATION_COUNT,
        }

    def _check_citation_exists(self, text: str) -> bool:
        """Check if any citation pattern exists in text."""
        for pattern in self.CITATION_PATTERNS:
            if re.search(pattern, text):
                return True
        return False

    def _extract_cited_files(self, text: str) -> set:
        """Extract filenames from citation markers."""
        cited = set()
        patterns = [
            r"\[Sumber:\s*([^,\]]+)",
            r"\[Source:\s*([^,\]]+)",
            r"Sumber:\s*([^.]+)\.",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            cited.update(m.strip() for m in matches)
        return cited


class RAGChain:
    """LangChain-based RAG chain with citation enforcement."""

    SYSTEM_PROMPT = """Anda adalah asisten AI yang menjawab pertanyaan HANYA berdasarkan dokumen yang diberikan.

ATURAN PENTING:
1. Jawab HANYA berdasarkan informasi dalam dokumen yang disediakan.
2. Jika informasi tidak ada di dokumen, katakan "Informasi ini tidak tersedia dalam dokumen yang diberikan."
3. SELALU sertakan sitasi/sumber dokumen di setiap klaim informasi menggunakan format:
   [Sumber: nama_file.pdf, Halaman X]
4. Jika informasi berasal dari beberapa dokumen, cantumkan SEMUA sumbernya.
5. Jangan mengarang informasi di luar dokumen.

Format jawaban:
- Mulai dengan jawaban singkat
- Sertakan sitasi di setiap kalimat yang merujuk dokumen
- Akhiri dengan ringkasan sumber yang digunakan"""

    USER_PROMPT = """Konteks dokumen:
{context}

Pertanyaan: {question}

Jawab dengan selalu menyertakan sumber/sitasi dokumen."""

    def __init__(self, retriever):
        self.retriever = retriever
        self.validator = CitationValidator()
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            temperature=0.1,
            max_tokens=2048,
        )
        self.chain = self._build_chain()

    def _build_chain(self):
        """Build the LangChain RAG chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", self.USER_PROMPT),
        ])

        def format_docs(docs):
            formatted = []
            for doc in docs:
                source = doc.metadata.get("filename", "unknown")
                page = doc.metadata.get("page", 0) + 1
                formatted.append(
                    f"[Dokumen: {source}, Halaman {page}]\n{doc.page_content}"
                )
            return "\n\n".join(formatted)

        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def query(self, question: str) -> Dict[str, Any]:
        """Execute query with citation validation and retry if needed."""
        MAX_RETRIES = 2

        for attempt in range(MAX_RETRIES):
            raw_response = self.chain.invoke(question)

            source_docs = self.retriever.invoke(question)
            validation = self.validator.validate(raw_response, source_docs)

            if validation["valid"]:
                return {
                    "answer": raw_response,
                    "sources": source_docs,
                    "validation": validation,
                    "attempts": attempt + 1,
                }

        retry_prompt = (
            f"Jawaban sebelumnya kurang lengkap dalam menyertakan sitasi.\n"
            f"Silakan jawab ulang pertanyaan berikut dengan format sitasi yang benar:\n"
            f"[Sumber: nama_file.pdf, Halaman X]\n\n"
            f"Pertanyaan: {question}"
        )
        final_response = self.chain.invoke(retry_prompt)
        source_docs = self.retriever.invoke(question)
        validation = self.validator.validate(final_response, source_docs)

        return {
            "answer": final_response,
            "sources": source_docs,
            "validation": validation,
            "attempts": MAX_RETRIES + 1,
        }
