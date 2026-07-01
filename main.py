"""CLI interface for Internal Q&A System."""

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table

from ingestion import DocumentIngestor
from chain import RAGChain
from config import DOCUMENTS_DIR

console = Console()


def print_header():
    console.print(
        Panel.fit(
            "[bold cyan]Internal Q&A System[/bold cyan]\n"
            "[dim]Powered by Mimo V2.5 + LangChain + ChromaDB[/dim]",
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

    table = Table(title="Ingestion Results")
    table.add_column("File", style="cyan")
    table.add_column("Chunks", style="green", justify="right")
    table.add_column("Status", style="yellow")

    for filename, count in results.items():
        status = "[green]OK[/green]" if count > 0 else "[red]FAILED[/red]"
        table.add_row(filename, str(count) if count > 0 else "-", status)

    console.print(table)
    console.print(f"\n[bold green]Total chunks created: {total_chunks}[/bold green]")


def query_command(chain: RAGChain, ingestor: DocumentIngestor):
    console.print("\n[bold cyan]Interactive Q&A Mode[/bold cyan]")
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
                "  help  - Show this help\n"
                "  stats - Show ingestion stats\n"
                "  exit  - Quit the program[/dim]"
            )
            continue

        if question.lower() == "stats":
            if ingestor.vectorstore is not None:
                collection = ingestor.vectorstore._collection
                console.print(f"\n[bold cyan]Statistik Ingestion:[/bold cyan]")
                console.print(f"  Collection: [green]{collection.name}[/green]")
                console.print(f"  Total chunks: [green]{collection.count()}[/green]")
                console.print(f"  Dokumen: [green]{DOCUMENTS_DIR}[/green]\n")
            else:
                console.print("[yellow]Belum ada data yang di-ingest. Jalankan 'python main.py ingest' terlebih dahulu.[/yellow]\n")
            continue

        with console.status("[bold cyan]Thinking...[/bold cyan]"):
            result = chain.query(question)

        console.print("\n[bold magenta]AI:[/bold magenta]")
        console.print(Markdown(result["answer"]))

        validation = result["validation"]
        if validation["valid"]:
            console.print(
                f"\n[green]Citation valid: {validation['citation_count']} source(s) referenced[/green]"
            )
        else:
            console.print(
                f"\n[yellow]Citation warning: {validation['citation_count']} source(s) found[/yellow]"
            )

        if result["attempts"] > 1:
            console.print(f"[dim]Answer generated in {result['attempts']} attempts (citation enforced)[/dim]")

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
            chain = RAGChain(ingestor.get_retriever())
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
        chain = RAGChain(ingestor.get_retriever())
        query_command(chain, ingestor)


if __name__ == "__main__":
    main()
