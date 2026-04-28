# tests/test_simple_router.py
import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.agent.nodes.simple_router import simple_router_node
from src.agent.state import PlannerOutput


class TestSimpleRouterNode:
    def _make_state(self, question="Qual a producao de soja?"):
        return {
            "question": question,
            "trace": [],
            "executed_agents": [],
            "token_usage": {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0},
        }

    def _mock_llm_response(self, datasource, tables=None, documents=None):
        content = json.dumps(
            {
                "datasource": datasource,
                "tables": tables or [],
                "documents": documents or [],
            }
        )
        return AIMessage(
            content=content,
            response_metadata={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        )

    @patch("src.agent.nodes.simple_router.get_node_llm")
    @patch("src.agent.nodes.simple_router.format_structured_map_summary", return_value="<struct_map/>")
    @patch("src.agent.nodes.simple_router.format_unstructured_map_to_context", return_value="<unstruct_map/>")
    def test_structured_routing(self, mock_unstruct, mock_struct, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_llm_response("structured", tables=["producao"])
        mock_get_llm.return_value = mock_llm

        result = simple_router_node(self._make_state())

        assert result["needs_retrieval"] is True
        planner_output = result["planner_output"]
        assert isinstance(planner_output, PlannerOutput)
        assert len(planner_output.searches) == 1
        assert planner_output.searches[0].type == "sql"
        assert planner_output.searches[0].query == "Qual a producao de soja?"
        assert planner_output.searches[0].sources == ["producao"]

    @patch("src.agent.nodes.simple_router.get_node_llm")
    @patch("src.agent.nodes.simple_router.format_structured_map_summary", return_value="<struct_map/>")
    @patch("src.agent.nodes.simple_router.format_unstructured_map_to_context", return_value="<unstruct_map/>")
    def test_non_structured_routing(self, mock_unstruct, mock_struct, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_llm_response("non_structured", documents=["manejo.pdf"])
        mock_get_llm.return_value = mock_llm

        result = simple_router_node(self._make_state("O que e manejo integrado?"))

        planner_output = result["planner_output"]
        assert len(planner_output.searches) == 1
        assert planner_output.searches[0].type == "text"
        assert planner_output.searches[0].sources == ["manejo.pdf"]

    @patch("src.agent.nodes.simple_router.get_node_llm")
    @patch("src.agent.nodes.simple_router.format_structured_map_summary", return_value="<struct_map/>")
    @patch("src.agent.nodes.simple_router.format_unstructured_map_to_context", return_value="<unstruct_map/>")
    def test_hybrid_routing(self, mock_unstruct, mock_struct, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_llm_response(
            "hybrid", tables=["producao"], documents=["relatorio.pdf"]
        )
        mock_get_llm.return_value = mock_llm

        result = simple_router_node(self._make_state())

        planner_output = result["planner_output"]
        assert len(planner_output.searches) == 2
        types = {s.type for s in planner_output.searches}
        assert types == {"sql", "text"}

    @patch("src.agent.nodes.simple_router.get_node_llm")
    @patch("src.agent.nodes.simple_router.format_structured_map_summary", return_value="<struct_map/>")
    @patch("src.agent.nodes.simple_router.format_unstructured_map_to_context", return_value="<unstruct_map/>")
    def test_original_question_used_as_query(self, mock_unstruct, mock_struct, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_llm_response("structured", tables=["producao"])
        mock_get_llm.return_value = mock_llm

        question = "Quais os 5 maiores produtores de soja?"
        result = simple_router_node(self._make_state(question))

        assert result["planner_output"].searches[0].query == question

    @patch("src.agent.nodes.simple_router.get_node_llm")
    @patch("src.agent.nodes.simple_router.format_structured_map_summary", return_value="<struct_map/>")
    @patch("src.agent.nodes.simple_router.format_unstructured_map_to_context", return_value="<unstruct_map/>")
    def test_prompt_contains_router_role(self, mock_unstruct, mock_struct, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_llm_response("structured", tables=["producao"])
        mock_get_llm.return_value = mock_llm

        simple_router_node(self._make_state())

        call_args = mock_llm.invoke.call_args[0][0]
        system_msg = call_args[0].content
        assert "ROTEADOR" in system_msg or "ROUTER" in system_msg

    @patch("src.agent.nodes.simple_router.get_node_llm")
    @patch("src.agent.nodes.simple_router.format_structured_map_summary", return_value="<struct_map/>")
    @patch("src.agent.nodes.simple_router.format_unstructured_map_to_context", return_value="<unstruct_map/>")
    def test_tracking_data_returned(self, mock_unstruct, mock_struct, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_llm_response("structured", tables=["producao"])
        mock_get_llm.return_value = mock_llm

        result = simple_router_node(self._make_state())

        assert "trace" in result
        assert "executed_agents" in result
        assert "token_usage" in result
