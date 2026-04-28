from unittest.mock import MagicMock, patch

from eval.judges import ResponseJudgeVerdict, SQLJudgeVerdict, judge_final_result, judge_sql_result


@patch("eval.judges._default_judge_llm")
def test_sql_judge_passes_question_to_prompt(mock_get_llm):
    """SQL judge must include the original question in the prompt."""
    mock_llm = MagicMock()
    mock_verdict = SQLJudgeVerdict(match=True, reasoning="equivalent")
    mock_llm.with_structured_output.return_value = mock_llm
    mock_llm.invoke.return_value = mock_verdict
    mock_get_llm.return_value = mock_llm

    output = {"sql_query": "SELECT * FROM t", "result_raw": "[{'a': 1}]"}
    input_data = {
        "question": "Quais os maiores produtores?",
        "sql_query": "SELECT * FROM t LIMIT 5",
        "sql_result": [{"a": 1}],
    }

    result = judge_sql_result(output, input_data)

    call_args = mock_llm.invoke.call_args[0][0]
    user_msg_content = call_args[1].content
    assert "Quais os maiores produtores?" in user_msg_content
    assert result["match"] is True


@patch("eval.judges._default_judge_llm")
def test_sql_judge_returns_dict(mock_get_llm):
    """SQL judge returns a dict with match and reasoning keys."""
    mock_llm = MagicMock()
    mock_verdict = SQLJudgeVerdict(match=False, reasoning="missing rows")
    mock_llm.with_structured_output.return_value = mock_llm
    mock_llm.invoke.return_value = mock_verdict
    mock_get_llm.return_value = mock_llm

    output = {"sql_query": "SELECT 1", "result_raw": "[]"}
    input_data = {"question": "Q?", "sql_query": "SELECT 1", "sql_result": [{"a": 1}]}

    result = judge_sql_result(output, input_data)

    assert "match" in result
    assert "reasoning" in result


@patch("eval.judges._default_judge_llm")
def test_response_judge_returns_all_dimensions(mock_get_llm):
    """Response judge returns completude, fidelidade, rastreabilidade and avg_score."""
    mock_llm = MagicMock()
    mock_verdict = ResponseJudgeVerdict(completude=2, fidelidade=1, rastreabilidade=2, reasoning="good")
    mock_llm.with_structured_output.return_value = mock_llm
    mock_llm.invoke.return_value = mock_verdict
    mock_get_llm.return_value = mock_llm

    output = {
        "final_answer": "A resposta completa.",
        "sql_results": [{"sql_query": "SELECT 1", "result_raw": "[{}]", "sources": ["t1"]}],
        "text_results": [{"sources": ["doc1.pdf"]}],
    }
    input_data = {"expected_answer": "Resposta esperada.", "question": "Pergunta?"}

    result = judge_final_result(output, input_data)

    assert result["completude"] == 2
    assert result["fidelidade"] == 1
    assert result["rastreabilidade"] == 2
    assert result["avg_score"] == round((2 + 1 + 2) / 3, 2)


@patch("eval.judges._default_judge_llm")
def test_response_judge_prompt_mentions_extra_info(mock_get_llm):
    """Response judge prompt must instruct that extra correct info does not penalize."""
    mock_llm = MagicMock()
    mock_verdict = ResponseJudgeVerdict(completude=2, fidelidade=2, rastreabilidade=2, reasoning="ok")
    mock_llm.with_structured_output.return_value = mock_llm
    mock_llm.invoke.return_value = mock_verdict
    mock_get_llm.return_value = mock_llm

    output = {"final_answer": "answer", "sql_results": [], "text_results": []}
    input_data = {"expected_answer": "expected", "question": "Q?"}

    judge_final_result(output, input_data)

    call_args = mock_llm.invoke.call_args[0][0]
    system_msg = call_args[0].content
    assert "additional" in system_msg.lower() or "extra" in system_msg.lower()
