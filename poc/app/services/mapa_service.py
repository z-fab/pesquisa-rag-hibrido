import json
from typing import Optional

from repositories.mapa_repository import (
    load_semantic_map_non_struct,
    load_semantic_map_struct,
)


def format_struct_semantic_map_to_context(
    tables: Optional[list[str]] = None, include_metrics: bool = False
) -> str:
    """Gera contexto compacto do schema apenas com descrição e colunas."""

    semantic_map = load_semantic_map_struct()
    selected = []
    for tb in semantic_map["tables"]:
        if tables and tb["table_name"] not in tables:
            continue
        selected.append(tb)
    if not selected:
        selected = semantic_map["tables"]

    lines: list[str] = []
    lines.append("<sql_schema>")

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
                        # Garante string legível, inclusive para listas/dicts
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


def format_non_struct_semantic_map_to_context(docs: Optional[list[str]] = None) -> str:
    """
    Gera contexto dos documentos em formato XML.

    Sempre exibe tudo que estiver disponível no semantic map:
    - file_id (como atributo id do <document>)
    - title
    - summary
    - key_topics (como lista de <topic>)
    - quaisquer outros campos são colocados em <metadata> como tags individuais.
    """

    semantic_map = load_semantic_map_non_struct()

    selected = []
    for doc in semantic_map["documents"]:
        if docs and doc.get("file_id") not in docs:
            continue
        selected.append(doc)

    if not selected:
        selected = semantic_map["documents"]

    lines: list[str] = []
    lines.append("<documents>")

    for doc in selected:
        file_id = doc.get("file_id", "")
        title = (doc.get("title") or "").strip()
        summary = (doc.get("summary") or "").strip()
        topics = doc.get("key_topics") or []

        lines.append(f'  <document id="{file_id}">')

        if title:
            lines.append(f"    <title>{title}</title>")

        if summary:
            lines.append(f"    <summary>{summary}</summary>")

        if topics:
            lines.append("    <topics>")
            for topic in topics:
                lines.append(f"      <topic>{topic}</topic>")
            lines.append("    </topics>")

        # Qualquer outro campo do YAML que não seja os principais acima
        main_keys = {"file_id", "title", "summary", "key_topics"}
        extra_keys = [k for k in doc.keys() if k not in main_keys]

        if extra_keys:
            lines.append("    <metadata>")
            for key in sorted(extra_keys):
                value = doc.get(key)
                if isinstance(value, (list, dict)):
                    value_str = json.dumps(value, ensure_ascii=False)
                else:
                    value_str = str(value)
                # o nome da tag é o nome do campo no YAML
                lines.append(f"      <{key}>{value_str}</{key}>")
            lines.append("    </metadata>")

        lines.append("  </document>")

    lines.append("</documents>")

    return "\n".join(lines)
