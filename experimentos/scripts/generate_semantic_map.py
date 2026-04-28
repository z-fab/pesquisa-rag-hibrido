import os

import numpy as np
import polars as pl
import yaml
from langchain_core.messages import HumanMessage
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from sqlalchemy import inspect
from src.config.providers import get_inference_llm
from src.config.settings import SETTINGS
from src.db.chromadb import get_vectorstore
from src.db.sqlite import get_engine
from src.utils.tracking import parse_llm_json

console = Console()

MAX_CHUNKS_DIRECT = 30


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def _calculate_column_stats(series: pl.Series) -> dict:
    """Calculates descriptive statistics for a polars Series."""
    stats = {}
    total = len(series)
    if total == 0:
        return {"status": "empty_table"}

    null_count = series.null_count()
    null_pct = round((null_count / total) * 100, 2)
    stats["row_count"] = total
    stats["null_percentage"] = f"{null_pct}%"

    clean = series.drop_nulls()

    if series.dtype.is_numeric():
        if len(clean) > 0:
            stats["min"] = float(clean.min())
            stats["max"] = float(clean.max())
            stats["mean"] = round(float(clean.mean()), 2)
            stats["std_dev"] = round(float(clean.std()), 2)
            stats["median"] = round(float(clean.median()), 2)
    else:
        clean_str = clean.cast(pl.Utf8)
        unique_count = clean_str.n_unique()
        stats["unique_values_count"] = unique_count

        if len(clean_str) > 0:
            top = clean_str.value_counts().sort("count", descending=True).head(10)
            stats["top_frequent_values"] = top.get_column(clean_str.name).to_list()

        if 0 < unique_count <= 20:
            stats["all_unique_values"] = clean_str.unique().sort().to_list()

    return stats


def generate_structured_map() -> None:
    """Generates structured semantic map from SQLite tables."""
    engine = get_engine()
    insp = inspect(engine)
    table_names = insp.get_table_names()

    if not table_names:
        console.print(
            Panel(
                "[red]No tables found in SQLite. Run 'rag ingest' first.[/red]",
                title="Warning",
            )
        )
        return

    llm = get_inference_llm(SETTINGS)
    tables_data = []

    for table_name in table_names:
        logger.info(f"Processing table: {table_name}")

        with engine.connect() as conn:
            df = pl.read_database(f"SELECT * FROM {table_name}", connection=conn)

        columns_info = []
        stats_context = []
        for col_name in df.columns:
            col_stats = _calculate_column_stats(df[col_name])
            col_type = str(df[col_name].dtype)
            columns_info.append(
                {
                    "name": col_name,
                    "type": col_type,
                    "statistics": col_stats,
                }
            )
            stats_context.append(f"  - {col_name} ({col_type}): {col_stats}")

        prompt = f"""Based on the following table schema and statistics, generate:
1. A concise description of what this table contains (1-2 sentences)
2. A concise description for each column (1 sentence each)

Table: {table_name}
Columns and statistics:
{chr(10).join(stats_context)}

Respond in JSON format:
{{"table_description": "...", "columns": {{"column_name": "description", ...}}}}"""

        response = llm.invoke([HumanMessage(content=prompt)])

        descriptions = parse_llm_json(response)
        if descriptions is None:
            descriptions = {"table_description": "", "columns": {}}

        table_entry = {
            "table_name": table_name,
            "description": descriptions.get("table_description", ""),
            "columns": [],
        }

        for col in columns_info:
            col["description"] = descriptions.get("columns", {}).get(col["name"], "")
            table_entry["columns"].append(col)

        tables_data.append(table_entry)

    output = {"tables": tables_data}
    with open(SETTINGS.PATH_STRUCTURED_MAP, "w", encoding="utf-8") as f:
        yaml.dump(
            output,
            f,
            Dumper=NoAliasDumper,
            sort_keys=False,
            allow_unicode=True,
            width=1000,
        )

    console.print(
        f"[green]Structured map saved to: {SETTINGS.PATH_STRUCTURED_MAP}[/green]"
    )


def generate_unstructured_map() -> None:
    """Generates unstructured semantic map from ChromaDB documents."""
    vectorstore = get_vectorstore()
    collection = vectorstore._collection

    if collection.count() == 0:
        console.print(
            Panel(
                "[red]No documents found in ChromaDB. Run 'rag ingest' first.[/red]",
                title="Warning",
            )
        )
        return

    all_data = collection.get(include=["metadatas", "documents", "embeddings"])
    metadatas = all_data.get("metadatas", [])
    documents = all_data.get("documents", [])
    embeddings = all_data.get("embeddings", [])

    source_groups: dict[str, list[dict]] = {}
    for i, meta in enumerate(metadatas):
        source = meta.get("source", "unknown")
        if source not in source_groups:
            source_groups[source] = []
        source_groups[source].append(
            {
                "content": documents[i] if i < len(documents) else "",
                "embedding": embeddings[i]
                if embeddings is not None and i < len(embeddings)
                else None,
            }
        )

    llm = get_inference_llm(SETTINGS)
    docs_data = []

    for source, chunks in source_groups.items():
        logger.info(f"Processing document: {source} ({len(chunks)} chunks)")

        if len(chunks) <= MAX_CHUNKS_DIRECT:
            selected_chunks = chunks
        else:
            valid_embeddings = [
                c["embedding"] for c in chunks if c["embedding"] is not None
            ]
            if valid_embeddings:
                emb_matrix = np.array(valid_embeddings)
                centroid = emb_matrix.mean(axis=0)
                distances = np.linalg.norm(emb_matrix - centroid, axis=1)
                nearest_indices = np.argsort(distances)[:MAX_CHUNKS_DIRECT]
                selected_chunks = [chunks[i] for i in nearest_indices]
            else:
                selected_chunks = chunks[:MAX_CHUNKS_DIRECT]

        sample_text = "\n---\n".join(
            [c["content"] for c in selected_chunks if c["content"]]
        )

        prompt = f"""Based on the following representative excerpts from a document, generate:
1. A title for this document
2. A concise summary (2-3 sentences)
3. A list of key topics covered
4. Any relevant metadata you can infer (year, institution, etc.)

Document source: {source}

Excerpts:
{sample_text[:8000]}

Respond in JSON format:
{{"title": "...", "summary": "...", "topics": ["topic1", ...], "metadata": {{"key": "value"}}}}"""

        response = llm.invoke([HumanMessage(content=prompt)])

        doc_info = parse_llm_json(response)
        if doc_info is None:
            doc_info = {"title": source, "summary": "", "topics": [], "metadata": {}}

        file_id = os.path.splitext(source)[0]
        doc_entry = {
            "file_id": file_id,
            "source": source,
            "title": doc_info.get("title", source),
            "summary": doc_info.get("summary", ""),
            "topics": doc_info.get("topics", []),
        }
        extra = doc_info.get("metadata", {})
        if extra:
            doc_entry.update(extra)

        docs_data.append(doc_entry)

    output = {"documents": docs_data}
    with open(SETTINGS.PATH_UNSTRUCTURED_MAP, "w", encoding="utf-8") as f:
        yaml.dump(
            output,
            f,
            Dumper=NoAliasDumper,
            sort_keys=False,
            allow_unicode=True,
            width=1000,
        )

    console.print(
        f"[green]Unstructured map saved to: {SETTINGS.PATH_UNSTRUCTURED_MAP}[/green]"
    )
