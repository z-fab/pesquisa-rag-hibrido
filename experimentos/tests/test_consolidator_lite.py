# tests/test_consolidator_lite.py
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.agent.nodes.consolidator_lite import consolidator_lite_node


class TestConsolidatorLiteNode:
    def _make_state(self, question="Qual a producao de soja?", sql_results=None, text_results=None):
        return {
            "question": question,
            "sql_results": sql_results or [],
            "text_results": text_results or [],
            "trace": [],
            "executed_agents": [],
            "token_usage": {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0},
        }

    def _mock_response(self, content="A producao de soja foi de 100 mil toneladas."):
        return AIMessage(
            content=content,
            response_metadata={"token_usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}},
        )

    @patch("src.agent.nodes.consolidator_lite.get_node_llm")
    def test_returns_final_answer_with_references(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_response("Resposta gerada.")
        mock_get_llm.return_value = mock_llm

        state = self._make_state(
            sql_results=[{"task_query": "q", "sql_query": "SELECT *", "result_raw": "100", "sources": ["producao"]}]
        )
        result = consolidator_lite_node(state)

        assert "Resposta gerada." in result["final_answer"]
        assert "Referências:" in result["final_answer"] or "Referencias:" in result["final_answer"]

    @patch("src.agent.nodes.consolidator_lite.get_node_llm")
    def test_sql_references_formatted(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_response("Dados SQL.")
        mock_get_llm.return_value = mock_llm

        state = self._make_state(sql_results=[{"sources": ["producao", "area_colhida"]}])
        result = consolidator_lite_node(state)

        assert "producao (SQL)" in result["final_answer"]
        assert "area_colhida (SQL)" in result["final_answer"]

    @patch("src.agent.nodes.consolidator_lite.get_node_llm")
    def test_text_references_formatted(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_response("Dados texto.")
        mock_get_llm.return_value = mock_llm

        state = self._make_state(text_results=[{"sources": ["manejo.pdf", "relatorio.pdf"]}])
        result = consolidator_lite_node(state)

        assert "manejo.pdf" in result["final_answer"]
        assert "relatorio.pdf" in result["final_answer"]

    @patch("src.agent.nodes.consolidator_lite.get_node_llm")
    def test_does_not_return_synthesizer_output(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_response("Resposta.")
        mock_get_llm.return_value = mock_llm

        result = consolidator_lite_node(self._make_state())

        assert "synthesizer_output" not in result

    @patch("src.agent.nodes.consolidator_lite.get_node_llm")
    def test_no_sources_no_references_section(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_response("Sem dados.")
        mock_get_llm.return_value = mock_llm

        result = consolidator_lite_node(self._make_state())

        assert result["final_answer"] == "Sem dados."

    @patch("src.agent.nodes.consolidator_lite.get_node_llm")
    def test_tracking_data_returned(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_response()
        mock_get_llm.return_value = mock_llm

        result = consolidator_lite_node(self._make_state())

        assert "trace" in result
        assert "executed_agents" in result
        assert "token_usage" in result
