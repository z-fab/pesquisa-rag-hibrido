from unittest.mock import patch

from typer.testing import CliRunner

from src.agent.ablation import AblationMode
from src.cli import app

runner = CliRunner()


class TestCLIAblation:
    @patch("src.cli._configure")
    @patch("src.cli.run_graph")
    def test_chat_default_ablation_is_full(self, mock_run_graph, mock_configure):
        mock_run_graph.return_value = {"final_answer": "test", "trace": [], "token_usage": {}, "total_start": 0}
        runner.invoke(app, ["chat", "test question"])
        mock_run_graph.assert_called_once_with("test question", mode=AblationMode.FULL)

    @patch("src.cli._configure")
    @patch("src.cli.run_graph")
    def test_chat_with_ablation_flag(self, mock_run_graph, mock_configure):
        mock_run_graph.return_value = {"final_answer": "test", "trace": [], "token_usage": {}, "total_start": 0}
        runner.invoke(app, ["chat", "test question", "--ablation", "poc"])
        mock_run_graph.assert_called_once_with("test question", mode=AblationMode.POC)

    @patch("src.cli._configure")
    @patch("src.cli.run_evaluation")
    def test_eval_default_ablation_is_full(self, mock_run_eval, mock_configure):
        mock_run_eval.return_value = ([], [], "test-run")
        runner.invoke(app, ["eval"])
        call_kwargs = mock_run_eval.call_args[1]
        assert call_kwargs["ablation_mode"] == AblationMode.FULL

    @patch("src.cli._configure")
    @patch("src.cli.run_evaluation")
    def test_eval_with_ablation_flag(self, mock_run_eval, mock_configure):
        mock_run_eval.return_value = ([], [], "test-run")
        runner.invoke(app, ["eval", "--ablation", "no-verifier"])
        call_kwargs = mock_run_eval.call_args[1]
        assert call_kwargs["ablation_mode"] == AblationMode.NO_VERIFIER
