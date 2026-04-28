from unittest.mock import MagicMock, patch

from src.agent.nodes.verifier import (
    _build_segments_context,
    _build_verifier_prompt,
    _invoke_verifier,
    verifier_node,
)
from src.agent.state import (
    CompletenessCheck,
    Reference,
    SegmentVerdict,
    SynthesizerOutput,
    SynthesizerSegment,
    VerifierOutput,
)


class TestVerifierModels:
    def test_segment_verdict(self):
        sv = SegmentVerdict(index=0, verdict="supported", reasoning="All claims backed by evidence.")
        assert sv.index == 0
        assert sv.verdict == "supported"
        assert sv.reasoning == "All claims backed by evidence."

    def test_segment_verdict_not_supported(self):
        sv = SegmentVerdict(index=1, verdict="not_supported", reasoning="No data found.")
        assert sv.verdict == "not_supported"

    def test_segment_verdict_partial(self):
        sv = SegmentVerdict(index=2, verdict="partial", reasoning="Some claims lack evidence.")
        assert sv.verdict == "partial"

    def test_completeness_check_covered(self):
        cc = CompletenessCheck(covered=True)
        assert cc.covered is True
        assert cc.missing_aspects == []

    def test_completeness_check_not_covered(self):
        cc = CompletenessCheck(covered=False, missing_aspects=["Faltou comparacao entre estados"])
        assert cc.covered is False
        assert len(cc.missing_aspects) == 1

    def test_verifier_output_pass(self):
        output = VerifierOutput(
            segments=[SegmentVerdict(index=0, verdict="supported", reasoning="OK")],
            completeness=CompletenessCheck(covered=True),
            overall_pass=True,
            feedback="",
        )
        assert output.overall_pass is True
        assert output.feedback == ""

    def test_verifier_output_fail(self):
        output = VerifierOutput(
            segments=[
                SegmentVerdict(index=0, verdict="supported", reasoning="OK"),
                SegmentVerdict(index=1, verdict="not_supported", reasoning="Dado inventado"),
            ],
            completeness=CompletenessCheck(covered=True),
            overall_pass=False,
            feedback="Segmento 2 contem dados nao suportados pela evidencia.",
        )
        assert output.overall_pass is False
        assert "Segmento 2" in output.feedback


class TestBuildSegmentsContext:
    def test_single_segment_with_refs(self):
        segments = [
            SynthesizerSegment(text="A soja e importante.", refs=[Reference(source="producao", type="sql")]),
        ]
        result = _build_segments_context(segments)
        assert '<segment index="0">' in result
        assert "<text>A soja e importante.</text>" in result
        assert "<source>producao</source>" in result
        assert "</segment>" in result

    def test_segment_without_refs(self):
        segments = [SynthesizerSegment(text="Conclusao geral.", refs=[])]
        result = _build_segments_context(segments)
        assert '<segment index="0">' in result
        assert "<text>Conclusao geral.</text>" in result

    def test_multiple_segments(self):
        segments = [
            SynthesizerSegment(text="Seg 1.", refs=[]),
            SynthesizerSegment(text="Seg 2.", refs=[Reference(source="a.pdf", type="text", section="Cap 1")]),
        ]
        result = _build_segments_context(segments)
        assert '<segment index="0">' in result
        assert '<segment index="1">' in result
        assert "<section>Cap 1</section>" in result


class TestBuildVerifierPrompt:
    def test_contains_all_sections(self):
        result = _build_verifier_prompt(
            question="Qual a producao de soja?",
            segments_xml="<segments/>",
            evidence_xml="<evidence/>",
        )
        assert "verification agent" in result
        assert "Qual a producao de soja?" in result
        assert "<segments/>" in result
        assert "<evidence/>" in result
        assert "overall_pass" in result
        assert "supported" in result.lower()
        assert "completeness" in result.lower()

    def test_contains_completeness_rubric(self):
        result = _build_verifier_prompt(
            question="Qual a producao de soja?",
            segments_xml="<segments/>",
            evidence_xml="<evidence/>",
        )
        assert "completeness_evaluation" in result
        assert "distinct aspect" in result
        assert "missing_aspects" in result
        assert "covered=false if ANY" in result


class TestInvokeVerifier:
    def test_structured_output_path(self):
        mock_parsed = VerifierOutput(
            segments=[SegmentVerdict(index=0, verdict="supported", reasoning="OK")],
            completeness=CompletenessCheck(covered=True),
            overall_pass=True,
            feedback="",
        )
        mock_raw = MagicMock()
        mock_raw.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = {"parsed": mock_parsed, "raw": mock_raw}
        result, usage = _invoke_verifier(mock_llm, [], VerifierOutput)
        assert result.overall_pass is True
        assert usage["input_tokens"] == 10

    def test_fallback_json_path(self):
        mock_response = MagicMock()
        mock_response.content = '{"segments": [{"index": 0, "verdict": "supported", "reasoning": "OK"}], "completeness": {"covered": true, "missing_aspects": []}, "overall_pass": true, "feedback": ""}'
        mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        result, usage = _invoke_verifier(mock_llm, [], VerifierOutput)
        assert result.overall_pass is True


class TestVerifierNode:
    @patch("src.agent.nodes.verifier._invoke_verifier")
    @patch("src.agent.nodes.verifier.get_node_llm")
    def test_passes_verification(self, mock_get_llm, mock_invoke):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_output = VerifierOutput(
            segments=[SegmentVerdict(index=0, verdict="supported", reasoning="OK")],
            completeness=CompletenessCheck(covered=True),
            overall_pass=True,
            feedback="",
        )
        mock_invoke.return_value = (mock_output, {"input_tokens": 50, "output_tokens": 30, "total_tokens": 80})
        state = {
            "question": "Qual a producao de soja?",
            "synthesizer_output": SynthesizerOutput(
                segments=[
                    SynthesizerSegment(text="MT produz 3800 kg/ha.", refs=[Reference(source="producao", type="sql")]),
                ]
            ),
            "sql_results": [
                {
                    "task_query": "Q",
                    "sql_query": "SELECT 1",
                    "result_raw": "1",
                    "sources": ["producao"],
                    "executed": True,
                    "error": "",
                },
            ],
            "text_results": [],
            "retry_count": 0,
        }
        result = verifier_node(state)
        output = result["verifier_output"]
        assert output.overall_pass is True
        assert output.feedback == ""
        assert result["retry_count"] == 0
        assert output.segments[0].verdict == "supported"
        assert "trace" in result
        assert result["trace"][0]["node"] == "verifier"

    @patch("src.agent.nodes.verifier._invoke_verifier")
    @patch("src.agent.nodes.verifier.get_node_llm")
    def test_fails_verification_increments_retry(self, mock_get_llm, mock_invoke):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_output = VerifierOutput(
            segments=[SegmentVerdict(index=0, verdict="not_supported", reasoning="Dado inventado")],
            completeness=CompletenessCheck(covered=True),
            overall_pass=False,
            feedback="Segmento 1 nao suportado.",
        )
        mock_invoke.return_value = (mock_output, {"input_tokens": 50, "output_tokens": 30, "total_tokens": 80})
        state = {
            "question": "Qual a producao?",
            "synthesizer_output": SynthesizerOutput(segments=[SynthesizerSegment(text="Dados inventados.", refs=[])]),
            "sql_results": [],
            "text_results": [],
            "retry_count": 0,
        }
        result = verifier_node(state)
        output = result["verifier_output"]
        assert output.overall_pass is False
        assert output.feedback == "Segmento 1 nao suportado."
        assert result["retry_count"] == 1

    @patch("src.agent.nodes.verifier._invoke_verifier")
    @patch("src.agent.nodes.verifier.get_node_llm")
    def test_passes_evidence_and_segments_to_prompt(self, mock_get_llm, mock_invoke):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_output = VerifierOutput(
            segments=[SegmentVerdict(index=0, verdict="supported", reasoning="OK")],
            completeness=CompletenessCheck(covered=True),
            overall_pass=True,
            feedback="",
        )
        mock_invoke.return_value = (mock_output, None)
        state = {
            "question": "Test?",
            "synthesizer_output": SynthesizerOutput(
                segments=[
                    SynthesizerSegment(text="Answer.", refs=[Reference(source="producao", type="sql")]),
                ]
            ),
            "sql_results": [
                {
                    "task_query": "Q",
                    "sql_query": "SELECT 1",
                    "result_raw": "1",
                    "sources": ["producao"],
                    "executed": True,
                    "error": "",
                },
            ],
            "text_results": [],
            "retry_count": 0,
        }
        verifier_node(state)
        mock_invoke.assert_called_once()
        call_args = mock_invoke.call_args
        messages = call_args[0][1]
        system_content = messages[0].content
        assert "sql_evidence" in system_content
        assert "producao" in system_content
        assert "Answer." in system_content
