import os

from docling.document_converter import DocumentConverter
from langchain_chroma import Chroma
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.config.providers import get_embeddings
from src.config.settings import SETTINGS

console = Console()


def ingest_unstructured() -> None:
    """Ingests PDF documents into ChromaDB using Docling + two-stage chunking."""
    pdf_folder = SETTINGS.PATH_DATA / "raw" / "unstructured"

    if not pdf_folder.exists():
        pdf_folder.mkdir(parents=True)
        console.print(f"[yellow]Created {pdf_folder}. Place your PDF files there and run again.[/yellow]")
        return

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        console.print("[yellow]No PDF files found in raw/unstructured/[/yellow]")
        return

    converter = DocumentConverter()
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    secondary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.INGEST_CHUNK_SIZE, chunk_overlap=SETTINGS.INGEST_CHUNK_OVERLAP
    )

    final_chunks = []
    summary_rows = []

    for filename in pdf_files:
        filepath = os.path.join(pdf_folder, filename)
        logger.info(f"Processing: {filename}")

        result = converter.convert(filepath)
        markdown_text = result.document.export_to_markdown()

        # Stage 1: split by headers
        header_splits = markdown_splitter.split_text(markdown_text)

        # Stage 2: split large chunks
        doc_chunks = []
        for split in header_splits:
            if len(split.page_content) > SETTINGS.INGEST_CHUNK_SIZE:
                sub_chunks = secondary_splitter.split_documents([split])
                for chunk in sub_chunks:
                    chunk.metadata["source"] = filename
                doc_chunks.extend(sub_chunks)
            else:
                split.metadata["source"] = filename
                doc_chunks.append(split)

        final_chunks.extend(doc_chunks)
        summary_rows.append((filename, len(doc_chunks)))

    if final_chunks:
        embeddings = get_embeddings(SETTINGS)
        Chroma.from_documents(
            documents=final_chunks,
            embedding=embeddings,
            persist_directory=str(SETTINGS.PATH_CHROMA_DB),
        )

    # Display summary
    table = Table(title="Unstructured Data Ingestion")
    table.add_column("Document", style="cyan")
    table.add_column("Chunks", justify="right", style="green")

    for name, count in summary_rows:
        table.add_row(name, str(count))

    table.add_row("[bold]Total[/bold]", f"[bold]{len(final_chunks)}[/bold]")
    console.print(table)


def ingest_structured() -> None:
    """Ingests structured data files (CSV) into SQLite using polars."""
    import polars as pl

    from src.db.sqlite import get_engine

    raw_folder = SETTINGS.PATH_DATA / "raw" / "structured"

    if not raw_folder.exists():
        raw_folder.mkdir(parents=True)
        console.print(f"[yellow]Created {raw_folder}. Place your CSV files there and run again.[/yellow]")
        return

    csv_files = [f for f in os.listdir(raw_folder) if f.endswith(".csv")]
    if not csv_files:
        console.print("[yellow]No CSV files found in raw/structured/[/yellow]")
        return

    engine = get_engine()
    summary_rows = []

    for filename in csv_files:
        filepath = os.path.join(raw_folder, filename)
        table_name = os.path.splitext(filename)[0]
        logger.info(f"Ingesting: {filename} -> table '{table_name}'")

        df = pl.read_csv(filepath)
        df.to_pandas().to_sql(table_name, engine, if_exists="replace", index=False)

        summary_rows.append((table_name, len(df)))

    # Display summary
    table = Table(title="Structured Data Ingestion")
    table.add_column("Table", style="cyan")
    table.add_column("Rows", justify="right", style="green")

    for name, count in summary_rows:
        table.add_row(name, str(count))

    console.print(table)
