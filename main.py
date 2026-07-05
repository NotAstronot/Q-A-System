"""CLI interface for Advanced RAG System."""

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table

from ingestion import DocumentIngestor
from retrieval import HybridRetriever, BM25Retriever, Reranker
from chain import RAGChain
from config import (
    DOCUMENTS_DIR,
    LLM_PROVIDER,
    RERANKER_MODEL,
    RERANKING_ENABLED,
    HYBRID_SEARCH_ENABLED,
    QUERY_REWRITING_ENABLED,
    TABLE_PARSING_ENABLED,
)

console = Console()


def print_header():
    features = []
    if HYBRID_SEARCH_ENABLED:
        features.append("Hybrid Search")
    if RERANKING_ENABLED:
        features.append("Reranker")
    if QUERY_REWRITING_ENABLED:
        features.append("Query Rewriting")
    if TABLE_PARSING_ENABLED:
        features.append("Table Parsing")

    console.print(
        Panel.fit(
            "[bold cyan]Advanced RAG System[/bold cyan]\n"
            f"[dim]Provider: {LLM_PROVIDER} | "
            f"Features: {', '.join(features)}[/dim]",
            border_style="cyan",
        )
    )


def ingest_command(ingestor: DocumentIngestor):
    console.print("\n[bold yellow]Ingesting documents...[/bold yellow]")

    if not any(DOCUMENTS_DIR.glob("*.pdf")):
        console.print(
            f"[red]No PDF files found in {DOCUMENTS_DIR}[/red]\n"
            f"[dim]Place your PDF files there and run again.[/dim]"
        )
        return

    results = ingestor.ingest_directory(str(DOCUMENTS_DIR))
    total_chunks = sum(v for v in results.values() if v > 0)

    table = Table(title="Ingestion Results (Parent-Child Chunking)")
    table.add_column("File", style="cyan")
    table.add_column("Child Chunks", style="green", justify="right")
    table.add_column("Status", style="yellow")

    for filename, count in results.items():
        status = "[green]OK[/green]" if count > 0 else "[red]FAILED[/red]"
        table.add_row(filename, str(count) if count > 0 else "-", status)

    console.print(table)
    console.print(f"\n[bold green]Total child chunks created: {total_chunks}[/bold green]")
    console.print("[dim]BM25 index built automatically.[/dim]")


def query_command(chain: RAGChain, ingestor: DocumentIngestor):
    console.print("\n[bold cyan]Interactive Q&A Mode[/bold cyan]")
    console.print(f"[dim]Provider: {LLM_PROVIDER}[/dim]")
    console.print("[dim]Type 'exit' to quit, 'help' for commands[/dim]\n")

    while True:
        try:
            question = Prompt.ask("[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if question.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if question.lower() == "help":
            console.print(
                "[dim]Commands:\n"
                "  help   - Show this help\n"
                "  stats  - Show ingestion stats\n"
                "  config - Show current configuration\n"
                "  exit   - Quit the program[/dim]"
            )
            continue

        if question.lower() == "stats":
            vectorstore = ingestor.get_vectorstore()
            collection = vectorstore._collection
            corpus, _ = ingestor.get_bm25_data()
            console.print(f"\n[bold cyan]System Statistics:[/bold cyan]")
            console.print(f"  Collection: [green]{collection.name}[/green]")
            console.print(f"  Total chunks: [green]{collection.count()}[/green]")
            console.print(f"  BM25 trained: [green]{len(corpus) > 0}[/green]")
            console.print(f"  Dokumen: [green]{DOCUMENTS_DIR}[/green]\n")
            continue

        if question.lower() == "config":
            console.print(f"\n[bold cyan]Configuration:[/bold cyan]")
            console.print(f"  Provider: [green]{LLM_PROVIDER}[/green]")
            console.print(f"  Hybrid Search: [green]{HYBRID_SEARCH_ENABLED}[/green]")
            console.print(f"  Reranker: [green]{RERANKING_ENABLED}[/green] ({RERANKER_MODEL})")
            console.print(f"  Query Rewriting: [green]{QUERY_REWRITING_ENABLED}[/green]")
            console.print(f"  Table Parsing: [green]{TABLE_PARSING_ENABLED}[/green]\n")
            continue

        with console.status("[bold cyan]Processing... (rewrite → search → rerank → generate → validate)"):
            result = chain.query(question)

        console.print("\n[bold magenta]AI:[/bold magenta]")
        console.print(Markdown(result["answer"]))

        if result.get("rewritten_query") and result["rewritten_query"] != question:
            console.print(f"\n[dim]Query rewritten: {result['rewritten_query']}[/dim]")

        validation = result["validation"]
        if validation["valid"]:
            console.print(
                f"[green]✓ Citation valid: {validation['citation_count']} source(s) referenced[/green]"
            )
        else:
            console.print(
                f"[yellow]⚠ Citation warning: {validation['citation_count']} source(s) found[/yellow]"
            )

        if result["attempts"] > 1:
            console.print(f"[dim]Generated in {result['attempts']} attempt(s)[/dim]")

        console.print()


def main():
    print_header()

    ingestor = DocumentIngestor()

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "ingest":
            ingest_command(ingestor)
            return
        elif command == "query":
            vectorstore = ingestor.get_vectorstore()
            corpus, metadata = ingestor.get_bm25_data()
            bm25 = BM25Retriever(corpus, metadata)
            hybrid = HybridRetriever(vectorstore, bm25)
            reranker = Reranker(RERANKER_MODEL)
            chain = RAGChain(hybrid, reranker)
            query_command(chain, ingestor)
            return
        else:
            console.print(f"[red]Unknown command: {command}[/red]")
            console.print("[dim]Usage: python main.py [ingest|query][/dim]")
            return

    console.print("\n[bold cyan]Commands:[/bold cyan]")
    console.print("  [green]python main.py ingest[/green]  - Ingest PDF documents")
    console.print("  [green]python main.py query[/green]   - Start interactive Q&A")

    choice = Prompt.ask(
        "\nSelect command",
        choices=["ingest", "query"],
        default="query",
    )

    if choice == "ingest":
        ingest_command(ingestor)
        console.print("\n[dim]Run 'python main.py query' to start asking questions.[/dim]")
    else:
        vectorstore = ingestor.get_vectorstore()
        corpus, metadata = ingestor.get_bm25_data()
        bm25 = BM25Retriever(corpus, metadata)
        hybrid = HybridRetriever(vectorstore, bm25)
        reranker = Reranker(RERANKER_MODEL)
        chain = RAGChain(hybrid, reranker)
        query_command(chain, ingestor)


if __name__ == "__main__":
    main()
