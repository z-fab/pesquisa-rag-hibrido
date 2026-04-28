from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.agent.nodes.sql_planner_executor import _execute_single_task, _extract_tables, _validate_query
from src.agent.state import SearchTask
from src.config.settings import Settings


def test_sql_max_retries_default():
    """Settings should have SQL_MAX_RETRIES with default value 3."""
    settings = Settings()
    assert settings.SQL_MAX_RETRIES == 3


class TestExtractTables:
    def test_single_table(self):
        assert _extract_tables("SELECT * FROM producao") == ["producao"]

    def test_multiple_tables_sorted(self):
        assert _extract_tables("SELECT * FROM producao JOIN estados ON producao.uf = estados.uf") == [
            "estados",
            "producao",
        ]

    def test_subquery(self):
        tables = _extract_tables("SELECT * FROM producao WHERE uf IN (SELECT uf FROM estados)")
        assert "estados" in tables
        assert "producao" in tables

    def test_invalid_sql_returns_empty(self):
        assert _extract_tables("NOT SQL") == []


class TestValidateQuery:
    def test_valid_select(self):
        is_valid, msg = _validate_query("SELECT * FROM users")
        assert is_valid is True
        assert msg == ""

    def test_valid_select_with_join(self):
        sql = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
        is_valid, msg = _validate_query(sql)
        assert is_valid is True

    def test_valid_cte(self):
        sql = "WITH top AS (SELECT id FROM users LIMIT 5) SELECT * FROM top"
        is_valid, msg = _validate_query(sql)
        assert is_valid is True

    def test_valid_subquery(self):
        sql = "SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)"
        is_valid, msg = _validate_query(sql)
        assert is_valid is True

    def test_valid_union(self):
        sql = "SELECT name FROM users UNION SELECT name FROM admins"
        is_valid, msg = _validate_query(sql)
        assert is_valid is True

    def test_column_named_update(self):
        """Columns with keywords like 'updated_at' must NOT be rejected."""
        sql = "SELECT updated_at, insert_date FROM logs"
        is_valid, msg = _validate_query(sql)
        assert is_valid is True

    def test_reject_update_statement(self):
        is_valid, msg = _validate_query("UPDATE users SET name = 'x'")
        assert is_valid is False
        assert "SELECT" in msg

    def test_reject_delete_statement(self):
        is_valid, msg = _validate_query("DELETE FROM users")
        assert is_valid is False

    def test_reject_insert_statement(self):
        is_valid, msg = _validate_query("INSERT INTO users VALUES (1, 'x')")
        assert is_valid is False

    def test_reject_drop_statement(self):
        is_valid, msg = _validate_query("DROP TABLE users")
        assert is_valid is False

    def test_reject_too_long(self):
        sql = "SELECT " + "a, " * 1000 + "b FROM t"
        is_valid, msg = _validate_query(sql)
        assert is_valid is False
        assert "too long" in msg.lower()

    def test_reject_malformed_sql(self):
        is_valid, msg = _validate_query("THIS IS NOT SQL AT ALL")
        assert is_valid is False

    def test_reject_empty_string(self):
        is_valid, msg = _validate_query("")
        assert is_valid is False


class TestExecuteSingleTask:
    def _make_llm(self, sql_response: str, usage: dict | None = None):
        token_usage = usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        msg = AIMessage(content=sql_response, response_metadata={"token_usage": token_usage})
        llm = MagicMock()
        llm.invoke.return_value = msg
        return llm

    @patch("src.agent.nodes.sql_planner_executor.format_structured_map_to_context", return_value="<schema/>")
    @patch("src.agent.nodes.sql_planner_executor.execute_query")
    def test_successful_execution(self, mock_exec, _mock_schema):
        mock_exec.return_value = [{"count": 42}]
        llm = self._make_llm("SELECT COUNT(*) as count FROM users")
        task = SearchTask(type="sql", query="How many users?", sources=["users"])

        result, usage = _execute_single_task(llm, task, max_retries=3)

        assert result["executed"] is True
        assert result["result"] == [{"count": 42}]
        assert result["sources"] == ["users"]  # extracted from SQL query AST
        assert result["error"] == ""
        assert usage["input_tokens"] > 0

    @patch("src.agent.nodes.sql_planner_executor.format_structured_map_to_context", return_value="<schema/>")
    @patch("src.agent.nodes.sql_planner_executor.execute_query")
    def test_retries_on_execution_error(self, mock_exec, _mock_schema):
        mock_exec.side_effect = [Exception("table locked"), [{"id": 1}]]
        llm = self._make_llm("SELECT id FROM users")
        task = SearchTask(type="sql", query="Get user ids", sources=["users"])

        result, usage = _execute_single_task(llm, task, max_retries=3)

        assert result["executed"] is True
        assert llm.invoke.call_count == 2

    @patch("src.agent.nodes.sql_planner_executor.format_structured_map_to_context", return_value="<schema/>")
    @patch("src.agent.nodes.sql_planner_executor.execute_query")
    def test_max_retries_exhausted(self, mock_exec, _mock_schema):
        mock_exec.side_effect = Exception("persistent error")
        llm = self._make_llm("SELECT bad FROM missing")
        task = SearchTask(type="sql", query="Bad query", sources=[])

        result, usage = _execute_single_task(llm, task, max_retries=2)

        assert result["executed"] is False
        assert result["sources"] == []  # no tables extracted on failure
        assert "persistent error" in result["error"]
        assert llm.invoke.call_count == 2
