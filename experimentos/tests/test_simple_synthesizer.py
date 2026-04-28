# tests/test_simple_synthesizer.py
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.agent.nodes.simple_synthesizer import simple_synthesizer_node


class TestSimpleSynthesizerNode:
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

    @patch("src.agent.nodes.simple_synthesizer.get_node_llm")
    def test_returns_final_answer(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_response("Resposta do modelo.")
        mock_get_llm.return_value = mock_llm

        result = simple_synthesizer_node(self._make_state())

        assert result["final_answer"] == "Resposta do modelo."

    @patch("src.agent.nodes.simple_synthesizer.get_node_llm")
    def test_does_not_return_synthesizer_output(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_response()
        mock_get_llm.return_value = mock_llm

        result = simple_synthesizer_node(self._make_state())

        assert "synthesizer_output" not in result

    @patch("src.agent.nodes.simple_synthesizer.get_node_llm")
    def test_prompt_contains_constraints(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_response()
        mock_get_llm.return_value = mock_llm

        simple_synthesizer_node(self._make_state())

        call_args = mock_llm.invoke.call_args[0][0]
        system_msg = call_args[0].content
        assert "MUST NOT" in system_msg or "NÃO" in system_msg
        assert "hallucinate" in system_msg.lower() or "alucinar" in system_msg.lower()

    @patch("src.agent.nodes.simple_synthesizer.get_node_llm")
    def test_prompt_does_not_contain_interpretive_analysis(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_response()
        mock_get_llm.return_value = mock_llm

        simple_synthesizer_node(self._make_state())

        call_args = mock_llm.invoke.call_args[0][0]
        system_msg = call_args[0].content
        assert "comparative analysis" not in system_msg

    @patch("src.agent.nodes.simple_synthesizer.get_node_llm")
    def test_tracking_data_returned(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_response()
        mock_get_llm.return_value = mock_llm

        result = simple_synthesizer_node(self._make_state())

        assert "trace" in result
        assert "executed_agents" in result
        assert "token_usage" in result

    @patch("src.agent.nodes.simple_synthesizer.get_node_llm")
    def test_evidence_included_in_prompt(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_response()
        mock_get_llm.return_value = mock_llm

        state = self._make_state(
            sql_results=[{"task_query": "producao", "sql_query": "SELECT *", "result_raw": "100", "sources": ["t1"]}]
        )
        simple_synthesizer_node(state)

        call_args = mock_llm.invoke.call_args[0][0]
        system_msg = call_args[0].content
        assert "producao" in system_msg or "evidence" in system_msg.lower()
