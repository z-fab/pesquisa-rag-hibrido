"""Tests for eval.report module."""

import json
import tempfile
from pathlib import Path

from rich.table import Table

from eval.report import build_summary_tables, load_results


def _make_results_json() -> dict:
    """Build a minimal valid results dict with 3 items: S1, NS1, H1."""
    return {
        "metadata": {
            "run_id": "test-run-001",
            "provider": "openai",
            "model": "gpt-4o",
            "timestamp": "2026-04-04T12:00:00",
        },
        "results": [
            {
                "id": "S1",
                "type": "S",
                "output_match_type": True,
                "output_type_predicted": "S",
                "output_sql_results": [{"col": "val"}],
                "output_rag": {"precision": None, "recall": None},
                "judgement": {
                    "sql": {"match": True, "reasoning": "Correct SQL"},
                    "response": {
                        "completude": 2,
                        "fidelidade": 2,
                        "rastreabilidade": 1,
                        "avg_score": 1.67,
                    },
                },
                "output_latency": {
                    "total": 5.2,
                    "per_agent": {"planner": 1.0, "sql_planner_executor": 4.2},
                },
                "output_token_usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                },
            },
            {
                "id": "NS1",
                "type": "NS",
                "output_match_type": True,
                "output_type_predicted": "NS",
                "output_sql_results": [],
                "output_rag": {"precision": 0.8, "recall": 0.6},
                "judgement": {
                    "sql": {"match": None, "reasoning": None},
                    "response": {
                        "completude": 1,
                        "fidelidade": 2,
                        "rastreabilidade": 2,
                        "avg_score": 1.67,
                    },
                },
                "output_latency": {
                    "total": 3.1,
                    "per_agent": {"planner": 0.8, "text_retriever": 2.3},
                },
                "output_token_usage": {
                    "input_tokens": 80,
                    "output_tokens": 40,
                    "total_tokens": 120,
                },
            },
            {
                "id": "H1",
                "type": "H",
                "output_match_type": False,
                "output_type_predicted": "S",
                "output_sql_results": [{"col": "val"}],
                "output_rag": {"precision": 0.7, "recall": 0.5},
                "judgement": {
                    "sql": {"match": False, "reasoning": "Wrong SQL"},
                    "response": {
                        "completude": 1,
                        "fidelidade": 1,
                        "rastreabilidade": 1,
                        "avg_score": 1.0,
                    },
                },
                "output_latency": {
                    "total": 8.0,
                    "per_agent": {
                        "planner": 1.2,
                        "sql_planner_executor": 3.5,
                        "text_retriever": 3.3,
                    },
                },
                "output_token_usage": {
                    "input_tokens": 200,
                    "output_tokens": 100,
                    "total_tokens": 300,
                },
            },
        ],
        "metrics": {
            "routing": {
                "accuracy": 66.67,
                "confusion_matrix": {
                    "S": {"S": 1, "NS": 0, "H": 0},
                    "NS": {"S": 0, "NS": 1, "H": 0},
                    "H": {"S": 1, "NS": 0, "H": 0},
                },
            },
            "retrieval": {
                "S": {"execution_accuracy": 100.0, "answer_accuracy": 80.0},
                "NS": {"execution_accuracy": None, "answer_accuracy": None, "precision": 0.8, "recall": 0.6},
                "H": {"execution_accuracy": 50.0, "answer_accuracy": 40.0, "precision": 0.7, "recall": 0.5},
            },
            "final_answer_quality": {
                "overall_avg": {
                    "completude": 1.33,
                    "fidelidade": 1.67,
                    "rastreabilidade": 1.33,
                    "media": 1.44,
                },
                "by_type": {
                    "S": {
                        "completude": 2.0,
                        "fidelidade": 2.0,
                        "rastreabilidade": 1.0,
                        "media": 1.67,
                    },
                    "NS": {
                        "completude": 1.0,
                        "fidelidade": 2.0,
                        "rastreabilidade": 2.0,
                        "media": 1.67,
                    },
                    "H": {
                        "completude": 1.0,
                        "fidelidade": 1.0,
                        "rastreabilidade": 1.0,
                        "media": 1.0,
                    },
                },
            },
            "efficiency": {
                "avg_total_latency": 5.43,
                "avg_agent_latency": {
                    "planner": 1.0,
                    "sql_planner_executor": 3.85,
                    "text_retriever": 2.8,
                },
                "avg_token_usage": {
                    "input_tokens": 127,
                    "output_tokens": 63,
                    "total_tokens": 190,
                },
            },
        },
    }


def test_load_results_from_file():
    """Write results JSON to a temp file, load it, and check metadata."""
    data = _make_results_json()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        tmp_path = Path(f.name)

    try:
        loaded = load_results(tmp_path)
        assert loaded["metadata"]["run_id"] == "test-run-001"
        assert loaded["metadata"]["provider"] == "openai"
        assert loaded["metadata"]["model"] == "gpt-4o"
        assert len(loaded["results"]) == 3
    finally:
        tmp_path.unlink()


def test_build_summary_tables_returns_expected_keys():
    """Check that build_summary_tables returns all expected table keys."""
    data = _make_results_json()
    tables = build_summary_tables(data)

    expected_keys = {
        "metadata",
        "routing",
        "confusion_matrix",
        "retrieval",
        "quality",
        "detail",
        "efficiency",
    }
    assert set(tables.keys()) == expected_keys

    for key, table in tables.items():
        assert isinstance(table, Table), f"tables['{key}'] is not a Rich Table"
