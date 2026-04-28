import json

from src.repositories.semantic_map_repository import (
    load_structured_map,
    load_unstructured_map,
)


def format_structured_map_summary() -> str:
    """Formats the structured semantic map as a summary (table names + descriptions only)."""
    semantic_map = load_structured_map()
    tables = semantic_map.get("tables", [])

    lines: list[str] = ["<sql_tables>"]
    for tb in tables:
        table_name = tb.get("table_name", "")
        table_desc = (tb.get("description") or "").strip()
        if table_desc:
            lines.append(f'  <table name="{table_name}">{table_desc}</table>')
        else:
            lines.append(f'  <table name="{table_name}" />')
    lines.append("</sql_tables>")
    return "\n".join(lines)


def format_structured_map_to_context(tables: list[str] | None = None, include_metrics: bool = False) -> str:
    """Formats the structured semantic map as XML for prompt injection."""
    semantic_map = load_structured_map()
    selected = []
    for tb in semantic_map.get("tables", []):
        if tables and tb["table_name"] not in tables:
            continue
        selected.append(tb)
    if not selected:
        selected = semantic_map.get("tables", [])

    lines: list[str] = ["<sql_schema>"]

    for tb in selected:
        table_name = tb.get("table_name", "")
        table_desc = (tb.get("description") or "").strip()

        lines.append(f'  <table name="{table_name}">')
        if table_desc:
            lines.append(f"    <description>{table_desc}</description>")

        lines.append("    <columns>")
        for col in tb.get("columns", []):
            col_name = col.get("name", "")
            col_type = col.get("type", "")
            col_desc = (col.get("description") or "").strip()

            lines.append(f'      <column name="{col_name}" type="{col_type}">')
            if col_desc:
                lines.append(f"        <description>{col_desc}</description>")

            if include_metrics:
                stats = col.get("statistics") or {}
                if stats:
                    lines.append("        <statistics>")
                    for key, value in stats.items():
                        if isinstance(value, (list, dict)):
                            value_str = json.dumps(value, ensure_ascii=False)
                        else:
                            value_str = str(value)
                        lines.append(f"          <{key}>{value_str}</{key}>")
                    lines.append("        </statistics>")

            lines.append("      </column>")
        lines.append("    </columns>")
        lines.append("  </table>")

    lines.append("</sql_schema>")
    return "\n".join(lines)


def format_unstructured_map_to_context(
    docs: list[str] | None = None,
) -> str:
    """Formats the unstructured semantic map as XML for prompt injection."""
    semantic_map = load_unstructured_map()

    selected = []
    for doc in semantic_map.get("documents", []):
        if docs and doc.get("source", doc.get("file_id")) not in docs:
            continue
        selected.append(doc)
    if not selected:
        selected = semantic_map.get("documents", [])

    lines: list[str] = ["<documents>"]

    for doc in selected:
        source = doc.get("source", doc.get("file_id", ""))
        title = (doc.get("title") or "").strip()
        summary = (doc.get("summary") or "").strip()
        topics = doc.get("key_topics") or doc.get("topics") or []

        lines.append(f'  <document id="{source}">')
        if title:
            lines.append(f"    <title>{title}</title>")
        if summary:
            lines.append(f"    <summary>{summary}</summary>")
        if topics:
            lines.append("    <topics>")
            for topic in topics:
                lines.append(f"      <topic>{topic}</topic>")
            lines.append("    </topics>")

        main_keys = {"file_id", "source", "title", "summary", "key_topics", "topics"}
        extra_keys = [k for k in doc if k not in main_keys]
        if extra_keys:
            lines.append("    <metadata>")
            for key in sorted(extra_keys):
                value = doc.get(key)
                value_str = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
                lines.append(f"      <{key}>{value_str}</{key}>")
            lines.append("    </metadata>")

        lines.append("  </document>")

    lines.append("</documents>")
    return "\n".join(lines)
