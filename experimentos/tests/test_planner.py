from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from src.agent.nodes.planner import _build_initial_prompt, _invoke_planner
from src.agent.state import PlannerOutput, SearchTask


def test_invoke_planner_structured_output():
    """When LLM returns structured output (dict with parsed key), extract directly."""
    expected = PlannerOutput(
        searches=[
            SearchTask(type="sql", query="total revenue 2024", sources=["revenue"]),
        ]
    )
    raw_msg = AIMessage(
        content="",
        response_metadata={
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        },
    )

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = {
        "raw": raw_msg,
        "parsed": expected,
        "parsing_error": None,
    }

    result, usage = _invoke_planner(mock_llm, [], PlannerOutput)

    assert result == expected
    assert usage["input_tokens"] == 10.0
    assert usage["output_tokens"] == 5.0


def test_invoke_planner_fallback_json():
    """When LLM returns AIMessage (no structured output), parse JSON from content."""
    json_content = '{"searches": [{"type": "text", "query": "strategic goals", "sources": ["report.pdf"]}]}'
    raw_msg = AIMessage(
        content=json_content,
        response_metadata={
            "token_usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            }
        },
    )

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = raw_msg

    result, usage = _invoke_planner(mock_llm, [], PlannerOutput)

    assert len(result.searches) == 1
    assert result.searches[0].type == "text"
    assert result.searches[0].query == "strategic goals"
    assert usage["input_tokens"] == 20.0


def test_invoke_planner_raises_on_invalid_json():
    """When LLM returns unparseable content, raise ValueError."""
    raw_msg = AIMessage(content="I don't know how to answer that", response_metadata={})

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = raw_msg

    with pytest.raises(ValueError, match="Failed to parse planner output"):
        _invoke_planner(mock_llm, [], PlannerOutput)


def test_invoke_planner_fallback_fenced_json():
    """When LLM returns JSON wrapped in markdown fences, parse correctly."""
    fenced = '```json\n{"searches": [{"type": "sql", "query": "count users", "sources": ["users"]}]}\n```'
    raw_msg = AIMessage(
        content=fenced,
        response_metadata={
            "token_usage": {
                "prompt_tokens": 15,
                "completion_tokens": 8,
                "total_tokens": 23,
            }
        },
    )

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = raw_msg

    result, usage = _invoke_planner(mock_llm, [], PlannerOutput)

    assert len(result.searches) == 1
    assert result.searches[0].type == "sql"
    assert result.searches[0].query == "count users"


def test_initial_prompt_contains_routing_guidelines():
    """Planner prompt must include routing guidelines with examples."""
    prompt = _build_initial_prompt()
    assert "routing_guidelines" in prompt
    assert "SQL ONLY" in prompt
    assert "TEXT ONLY" in prompt
    assert "CONNECTING quantitative data" in prompt
    # Check few-shot examples exist
    assert "temporal aggregation" in prompt
    assert "definition/concept" in prompt
    assert "RELATING quantitative evolution" in prompt


def test_initial_prompt_no_contradictory_routing_rule():
    """The old 'Do NOT add a search item just to cover both types' rule should be removed."""
    prompt = _build_initial_prompt()
    assert "Do NOT add a search item just to cover both types" not in prompt
