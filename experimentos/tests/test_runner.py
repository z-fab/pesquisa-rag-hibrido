import tempfile
from pathlib import Path

from eval.runner import _build_run_id, _load_checkpoint, _save_checkpoint_item


def test_build_run_id_with_explicit_label():
    assert _build_run_id("my-run") == "my-run"


def test_build_run_id_auto_generated():
    run_id = _build_run_id(None, provider="openai", model="gpt-4o-mini", ablation_mode="full")
    assert run_id.startswith("openai_gpt-4o-mini_full_")


def test_build_run_id_sanitizes_slash_in_model_name():
    """Model names like 'meta-llama/llama-3.1-8b' must not produce directory separators in the run_id."""
    run_id = _build_run_id(
        None, provider="openrouter", model="meta-llama/llama-3.1-8b-instruct", ablation_mode="full"
    )
    assert "/" not in run_id
    assert "openrouter_meta-llama--llama-3.1-8b-instruct_full_" in run_id


def test_build_run_id_preserves_valid_chars():
    run_id = _build_run_id(None, provider="groq", model="qwen-2.5-32b", ablation_mode="no-verifier")
    assert run_id.startswith("groq_qwen-2.5-32b_no-verifier_")


def test_save_and_load_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "checkpoint_test.json"
        _save_checkpoint_item(
            path=path,
            run_id="test-run",
            metadata={"provider": "openai"},
            item_id="S1",
            output={"final_answer": "answer"},
            evaluation={"id": "S1", "type": "S"},
            status="completed",
        )
        checkpoint = _load_checkpoint(path)
        assert checkpoint["run_id"] == "test-run"
        assert "S1" in checkpoint["completed"]
        assert checkpoint["completed"]["S1"]["evaluation"]["id"] == "S1"


def test_load_checkpoint_returns_empty_if_missing():
    path = Path("/nonexistent/checkpoint.json")
    checkpoint = _load_checkpoint(path)
    assert checkpoint["completed"] == {}
    assert checkpoint["failed"] == {}


def test_save_checkpoint_item_failed():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "checkpoint_test.json"
        _save_checkpoint_item(
            path=path,
            run_id="test-run",
            metadata={},
            item_id="S3",
            output=None,
            evaluation=None,
            status="failed",
            error="Connection timeout",
        )
        checkpoint = _load_checkpoint(path)
        assert "S3" in checkpoint["failed"]
        assert checkpoint["failed"]["S3"]["error"] == "Connection timeout"
