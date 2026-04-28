def build_evidence_context(state: dict) -> str:
    """Builds XML evidence context from SQL and text results in the agent state.

    Used by both the synthesizer (to provide context for answer generation)
    and the verifier (to check faithfulness of the generated answer).
    """
    sections = []

    sql_results = state.get("sql_results", [])
    for i, res in enumerate(sql_results):
        lines = [f'<sql_evidence index="{i + 1}">']
        lines.append(f"  <task_query>{res.get('task_query', 'N/A')}</task_query>")
        lines.append(f"  <sql_query>{res.get('sql_query', 'N/A')}</sql_query>")
        lines.append(f"  <tables>{', '.join(res.get('sources', []))}</tables>")
        lines.append(f"  <result>{res.get('result_raw', 'N/A')}</result>")
        lines.append("</sql_evidence>")
        sections.append("\n".join(lines))

    text_results = state.get("text_results", [])
    for i, res in enumerate(text_results):
        lines = [f'<text_evidence index="{i + 1}">']
        lines.append(f"  <task_query>{res.get('task_query', 'N/A')}</task_query>")
        for chunk in res.get("chunks", []):
            lines.append("  <chunk>")
            for key, value in chunk.items():
                lines.append(f"    <{key}>{value}</{key}>")
            lines.append("  </chunk>")
        lines.append("</text_evidence>")
        sections.append("\n".join(lines))

    return "\n".join(sections) if sections else "No evidence available."
