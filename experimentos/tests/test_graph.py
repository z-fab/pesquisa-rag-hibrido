from src.agent.ablation import AblationMode
from src.agent.graph import build_graph


class TestBuildGraph:
    def test_full_mode_has_all_nodes(self):
        graph = build_graph(AblationMode.FULL)
        node_names = set(graph.get_graph().nodes.keys())
        assert "planner" in node_names
        assert "sql_planner_executor" in node_names
        assert "text_retriever" in node_names
        assert "synthesizer" in node_names
        assert "verifier" in node_names
        assert "consolidator" in node_names

    def test_no_verifier_mode_excludes_verifier(self):
        graph = build_graph(AblationMode.NO_VERIFIER)
        node_names = set(graph.get_graph().nodes.keys())
        assert "planner" in node_names
        assert "synthesizer" in node_names
        assert "consolidator" in node_names
        assert "verifier" not in node_names

    def test_no_synthesizer_mode_uses_consolidator_lite(self):
        graph = build_graph(AblationMode.NO_SYNTHESIZER)
        node_names = set(graph.get_graph().nodes.keys())
        assert "planner" in node_names
        assert "consolidator_lite" in node_names
        assert "synthesizer" not in node_names
        assert "verifier" not in node_names
        assert "consolidator" not in node_names

    def test_poc_mode_uses_simple_nodes(self):
        graph = build_graph(AblationMode.POC)
        node_names = set(graph.get_graph().nodes.keys())
        assert "simple_router" in node_names
        assert "simple_synthesizer" in node_names
        assert "planner" not in node_names
        assert "verifier" not in node_names
        assert "consolidator" not in node_names

    def test_default_mode_is_full(self):
        graph_default = build_graph()
        graph_full = build_graph(AblationMode.FULL)
        default_nodes = set(graph_default.get_graph().nodes.keys())
        full_nodes = set(graph_full.get_graph().nodes.keys())
        assert default_nodes == full_nodes

    def test_all_modes_have_sql_and_text_nodes(self):
        for mode in AblationMode:
            graph = build_graph(mode)
            node_names = set(graph.get_graph().nodes.keys())
            assert "sql_planner_executor" in node_names, f"{mode} missing sql_planner_executor"
            assert "text_retriever" in node_names, f"{mode} missing text_retriever"
