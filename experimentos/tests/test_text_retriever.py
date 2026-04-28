from unittest.mock import MagicMock, patch

from src.agent.nodes.text_retriever import _execute_single_task, text_retriever_node
from src.agent.state import PlannerOutput, SearchTask
from src.config.settings import Settings


class TestSimilaritySearch:
    def _make_mock_doc(self, content: str, metadata: dict):
        doc = MagicMock()
        doc.page_content = content
        doc.metadata = metadata
        return doc

    @patch("src.repositories.chromadb_repository.get_vectorstore")
    def test_returns_all_metadata(self, mock_get_vs):
        from src.repositories.chromadb_repository import similarity_search

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [
            self._make_mock_doc(
                "chunk text",
                {
                    "source": "report.pdf",
                    "Header 1": "Chapter 1",
                    "Header 2": "Section A",
                },
            )
        ]
        mock_get_vs.return_value = mock_vs

        results = similarity_search("test query", k=3)

        assert len(results) == 1
        assert results[0]["content"] == "chunk text"
        assert results[0]["source"] == "report.pdf"
        assert results[0]["Header 1"] == "Chapter 1"
        assert results[0]["Header 2"] == "Section A"

    @patch("src.repositories.chromadb_repository.get_vectorstore")
    def test_passes_filter_and_k(self, mock_get_vs):
        from src.repositories.chromadb_repository import similarity_search

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        mock_get_vs.return_value = mock_vs

        similarity_search("q", k=10, filter_dict={"source": {"$in": ["a.pdf"]}})

        mock_vs.similarity_search.assert_called_once_with("q", k=10, filter={"source": {"$in": ["a.pdf"]}})


def test_text_search_k_default():
    """Settings should have TEXT_SEARCH_K with default value 5."""
    settings = Settings()
    assert settings.TEXT_SEARCH_K == 5


class TestExecuteSingleTask:
    @patch("src.agent.nodes.text_retriever.similarity_search")
    def test_returns_dict_chunks(self, mock_search):
        mock_search.return_value = [
            {"content": "chunk one", "source": "doc.pdf", "Header 1": "Intro"},
            {"content": "chunk two", "source": "doc.pdf"},
        ]
        task = SearchTask(type="text", query="What is X?", sources=["doc.pdf"])

        result = _execute_single_task(task, k=5)

        assert result["task_query"] == "What is X?"
        assert len(result["chunks"]) == 2
        assert result["chunks"][0]["content"] == "chunk one"
        assert result["chunks"][0]["source"] == "doc.pdf"
        assert result["chunks"][0]["Header 1"] == "Intro"
        assert result["chunks"][1]["content"] == "chunk two"
        assert "doc.pdf" in result["sources"]

    @patch("src.agent.nodes.text_retriever.similarity_search")
    def test_applies_source_filter(self, mock_search):
        mock_search.return_value = []
        task = SearchTask(type="text", query="Q", sources=["a.pdf", "b.pdf"])

        _execute_single_task(task, k=3)

        mock_search.assert_called_once_with(query="Q", k=3, filter_dict={"source": {"$in": ["a.pdf", "b.pdf"]}})

    @patch("src.agent.nodes.text_retriever.similarity_search")
    def test_no_filter_when_no_sources(self, mock_search):
        mock_search.return_value = []
        task = SearchTask(type="text", query="Q", sources=[])

        _execute_single_task(task, k=5)

        mock_search.assert_called_once_with(query="Q", k=5, filter_dict=None)

    @patch("src.agent.nodes.text_retriever.similarity_search")
    def test_unique_sources_in_result(self, mock_search):
        mock_search.return_value = [
            {"content": "a", "source": "doc.pdf"},
            {"content": "b", "source": "doc.pdf"},
            {"content": "c", "source": "other.pdf"},
        ]
        task = SearchTask(type="text", query="Q", sources=[])

        result = _execute_single_task(task, k=5)

        assert sorted(result["sources"]) == ["doc.pdf", "other.pdf"]


class TestTextRetrieverNode:
    @patch("src.agent.nodes.text_retriever.similarity_search")
    @patch("src.agent.nodes.text_retriever.SETTINGS")
    def test_uses_settings_k_and_embedding_tracking(self, mock_settings, mock_search):
        mock_settings.TEXT_SEARCH_K = 7
        mock_settings.EMBEDDING_PROVIDER = "openai"
        mock_settings.EMBEDDING_MODEL = "text-embedding-3-small"

        mock_search.return_value = [{"content": "test", "source": "f.pdf"}]

        state = {
            "question": "test?",
            "planner_output": PlannerOutput(searches=[SearchTask(type="text", query="Q", sources=[])]),
        }

        result = text_retriever_node(state)

        assert len(result["text_results"]) == 1
        assert result["trace"][0]["provider"] == "openai"
        assert result["trace"][0]["model"] == "text-embedding-3-small"
        mock_search.assert_called_once_with(query="Q", k=7, filter_dict=None)

    @patch("src.agent.nodes.text_retriever.similarity_search")
    @patch("src.agent.nodes.text_retriever.SETTINGS")
    def test_filters_only_text_tasks(self, mock_settings, mock_search):
        mock_settings.TEXT_SEARCH_K = 5
        mock_settings.EMBEDDING_PROVIDER = "openai"
        mock_settings.EMBEDDING_MODEL = "embed"

        mock_search.return_value = []

        state = {
            "question": "test?",
            "planner_output": PlannerOutput(
                searches=[
                    SearchTask(type="sql", query="SQL query", sources=["table1"]),
                    SearchTask(type="text", query="Text query", sources=["doc.pdf"]),
                ]
            ),
        }

        result = text_retriever_node(state)

        assert len(result["text_results"]) == 1
        assert mock_search.call_count == 1
