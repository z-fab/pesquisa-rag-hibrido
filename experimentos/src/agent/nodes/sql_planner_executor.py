import time

import sqlglot
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from sqlglot import exp as sqlglot_exp

from src.agent.state import AgentState, SearchTask
from src.config.providers import get_node_llm
from src.config.settings import SETTINGS
from src.repositories.sqlite_repository import execute_query
from src.services.semantic_map_service import format_structured_map_to_context
from src.utils.tracking import extract_usage_from_response, get_text_content, normalize_usage, record_end


def _validate_query(query: str) -> tuple[bool, str]:
    """Validates a SQL query using AST parsing with sqlglot."""
    if len(query) > 2000:
        return False, "Query too long (max 2000 chars)"
    try:
        statements = sqlglot.parse(query, dialect="sqlite")
    except sqlglot.errors.ParseError as e:
        return False, f"SQL parse error: {e}"
    if len(statements) != 1:
        return False, "Expected exactly one SQL statement"
    stmt = statements[0]
    if stmt is None:
        return False, "Empty or unparseable SQL"
    if not isinstance(stmt, (sqlglot_exp.Select, sqlglot_exp.Union)):
        return False, "Only SELECT queries are allowed"
    return True, ""


def _generate_sql(llm, question: str, tables: list[str] | None, error_context: str = "") -> tuple[str, dict | None]:
    """Generates SQL from a natural language question using the LLM. Returns (sql_string, usage)."""
    error_section = ""
    if error_context:
        error_section = f"""
<previous_error>
    The previous SQL query failed with this error:
    {error_context}
    Please generate a corrected query.
</previous_error>"""

    system_prompt = f"""<system>
<role>
    You are a SQL expert. Generate ONE valid SQLite query to answer the question below.
</role>

<rules>
    - Use ONLY tables and columns from the schema below.
    - NEVER invent tables or columns.
    - For text comparisons, use LOWER(column) LIKE '%text%'.
    - Use aggregate functions (SUM, AVG, MAX, MIN, COUNT) when appropriate.
    - If the question cannot be answered with the available schema, return:
      SELECT 'ERROR: QUERY NOT SUPPORTED BY SCHEMA' AS message;
    - Return ONLY the SQL query, no explanations, no markdown.
</rules>

{format_structured_map_to_context(tables, include_metrics=True)}
{error_section}
</system>"""

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"<question>{question}</question>"),
        ]
    )

    sql = get_text_content(response).replace("```sql", "").replace("```", "").strip()
    usage = extract_usage_from_response(response)
    return sql, usage


def _extract_tables(sql: str) -> list[str]:
    """Extracts table names from a SQL query using sqlglot AST parsing."""
    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
        return sorted({t.name for t in parsed.find_all(sqlglot_exp.Table)})
    except Exception:
        return []


def _execute_single_task(llm, task: SearchTask, max_retries: int) -> tuple[dict, dict]:
    """Executes a single SQL task with retry loop. Returns (result_entry, accumulated_usage)."""
    total_usage = {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0}
    last_error = ""
    result_entry = {
        "task_query": task.query,
        "sources": [],
        "sql_query": "",
        "executed": False,
        "result": [],
        "result_raw": "",
        "error": "",
    }

    logger.debug(f"Starting SQL generation for task: {task.query} with sources: {task.sources}")

    for attempt in range(max_retries):
        sql, usage = _generate_sql(llm, task.query, task.sources or None, last_error)
        if usage:
            norm = normalize_usage(usage)
            for k in total_usage:
                total_usage[k] += norm[k]

        result_entry["sql_query"] = sql

        is_valid, validation_msg = _validate_query(sql)
        if not is_valid:
            last_error = f"Validation error: {validation_msg}"
            logger.warning(f"SQL validation failed (attempt {attempt + 1}): {validation_msg}")
            continue

        try:
            rows = execute_query(sql)
            result_entry["executed"] = True
            result_entry["result"] = rows
            result_entry["result_raw"] = str(rows) if rows else "Query executed successfully, no results."
            result_entry["sources"] = _extract_tables(sql)
            break
        except Exception as e:
            last_error = str(e)
            logger.warning(f"SQL execution failed (attempt {attempt + 1}): {e}")

    if not result_entry["executed"]:
        result_entry["error"] = last_error

    logger.debug(
        f"Finished SQL task: {task.query} with result: {result_entry['result_raw']} and error: {result_entry['error']}"
    )

    return result_entry, total_usage


def sql_planner_executor_node(state: AgentState) -> dict:
    """SQL node: generates and executes SQL for each sql SearchTask with retry."""
    logger.info("Executing SQL Planner & Executor...")

    llm = get_node_llm("sql", SETTINGS)
    node_start = time.perf_counter()

    tasks = [t for t in state["planner_output"].searches if t.type == "sql"]
    all_results: list[dict] = []
    total_usage = {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0}

    for task in tasks:
        result, usage = _execute_single_task(llm, task, SETTINGS.SQL_MAX_RETRIES)
        all_results.append(result)
        for k in total_usage:
            total_usage[k] += usage[k]

    provider = SETTINGS.PROVIDER
    model_name = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
    tracking = record_end("sql_planner_executor", provider, model_name, node_start, total_usage)

    return {
        "sql_results": all_results,
        **tracking,
    }
