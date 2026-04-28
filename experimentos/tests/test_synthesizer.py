from unittest.mock import MagicMock, patch

from src.agent.nodes.synthesizer import (
    _build_initial_prompt,
    _build_revision_prompt,
    _invoke_synthesizer,
    synthesizer_node,
)
from src.agent.state import (
    CompletenessCheck,
    PlannerOutput,
    Reference,
    SearchTask,
    SegmentVerdict,
    SynthesizerOutput,
    SynthesizerSegment,
    VerifierOutput,
)


class TestSynthesizerModels:
    def test_reference_with_section(self):
        ref = Reference(source="doc.pdf", type="text", section="Capitulo 1")
        assert ref.source == "doc.pdf"
        assert ref.type == "text"
        assert ref.section == "Capitulo 1"

    def test_reference_without_section(self):
        ref = Reference(source="producao", type="sql")
        assert ref.section == ""

    def test_segment_with_refs(self):
        seg = SynthesizerSegment(
            text="A soja e a principal cultura.",
            refs=[Reference(source="producao", type="sql")],
        )
        assert seg.text == "A soja e a principal cultura."
        assert len(seg.refs) == 1

    def test_segment_without_refs(self):
        seg = SynthesizerSegment(text="Texto sem dados.")
        assert seg.refs == []

    def test_synthesizer_output(self):
        output = SynthesizerOutput(
            segments=[
                SynthesizerSegment(
                    text="Trecho um.",
                    refs=[Reference(source="doc.pdf", type="text", section="Cap 1")],
                ),
                SynthesizerSegment(text="Trecho dois."),
            ]
        )
        assert len(output.segments) == 2
        assert output.segments[0].refs[0].source == "doc.pdf"
        assert output.segments[1].refs == []


class TestInvokeSynthesizer:
    def test_structured_output_path(self):
        mock_parsed = SynthesizerOutput(
            segments=[SynthesizerSegment(text="Answer.", refs=[Reference(source="doc.pdf", type="text")])]
        )
        mock_raw = MagicMock()
        mock_raw.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = {"parsed": mock_parsed, "raw": mock_raw}

        result, usage = _invoke_synthesizer(mock_llm, [], SynthesizerOutput)

        assert len(result.segments) == 1
        assert result.segments[0].text == "Answer."
        assert usage["input_tokens"] == 10

    def test_fallback_json_path(self):
        mock_response = MagicMock()
        mock_response.content = '{"segments": [{"text": "Answer.", "refs": []}]}'
        mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        result, usage = _invoke_synthesizer(mock_llm, [], SynthesizerOutput)

        assert len(result.segments) == 1
        assert result.segments[0].text == "Answer."


class TestSynthesizerNode:
    @patch("src.agent.nodes.synthesizer._invoke_synthesizer")
    @patch("src.agent.nodes.synthesizer.get_node_llm")
    def test_returns_output_and_tracking(self, mock_get_llm, mock_invoke):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        mock_output = SynthesizerOutput(
            segments=[
                SynthesizerSegment(
                    text="A soja e importante.",
                    refs=[Reference(source="producao", type="sql")],
                ),
                SynthesizerSegment(text="Conclusao geral."),
            ]
        )
        mock_invoke.return_value = (mock_output, {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})

        state = {
            "question": "Qual a importancia da soja?",
            "planner_output": PlannerOutput(searches=[SearchTask(type="text", query="Q", sources=[])]),
            "sql_results": [],
            "text_results": [
                {"task_query": "Q", "chunks": [{"content": "c", "source": "a.pdf"}], "sources": ["a.pdf"]}
            ],
        }

        result = synthesizer_node(state)

        output = result["synthesizer_output"]
        assert len(output.segments) == 2
        assert output.segments[0].text == "A soja e importante."
        assert output.segments[0].refs[0].source == "producao"
        assert output.segments[1].refs == []
        assert "trace" in result
        assert result["trace"][0]["node"] == "synthesizer"

    @patch("src.agent.nodes.synthesizer._invoke_synthesizer")
    @patch("src.agent.nodes.synthesizer.get_node_llm")
    def test_passes_evidence_to_llm(self, mock_get_llm, mock_invoke):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        mock_output = SynthesizerOutput(segments=[SynthesizerSegment(text="Ok.")])
        mock_invoke.return_value = (mock_output, None)

        state = {
            "question": "Test?",
            "planner_output": PlannerOutput(searches=[]),
            "sql_results": [
                {"task_query": "Q", "sql_query": "SELECT 1", "result_raw": "1", "executed": True, "error": ""},
            ],
            "text_results": [],
        }

        synthesizer_node(state)

        mock_invoke.assert_called_once()
        call_args = mock_invoke.call_args
        messages = call_args[0][1]
        system_content = messages[0].content
        assert "sql_evidence" in system_content


class TestSynthesizerPromptBuilders:
    def test_initial_prompt_contains_rules(self):
        prompt = _build_initial_prompt("<evidence/>")
        assert "MUST NOT hallucinate or fabricate" in prompt
        assert "MAY derive comparisons, trends, and rankings" in prompt
        assert "MUST NOT speculate about causes" in prompt
        assert "MUST NOT use external knowledge" in prompt
        assert "Be thorough" in prompt
        assert "comparative analysis" in prompt
        assert "synthesis agent" in prompt
        assert "REVISION" not in prompt

    def test_revision_prompt_contains_rules_and_feedback(self):
        previous_segments = [
            SynthesizerSegment(text="Seg OK.", refs=[Reference(source="producao", type="sql")]),
            SynthesizerSegment(text="Seg ruim.", refs=[]),
        ]
        feedback = "Segmento 2 nao suportado."
        prompt = _build_revision_prompt("<evidence/>", previous_segments, feedback)
        assert "MUST NOT hallucinate or fabricate" in prompt
        assert "MAY derive comparisons, trends, and rankings" in prompt
        assert "MUST NOT use external knowledge" in prompt
        assert "REVISION" in prompt
        assert "Segmento 2 nao suportado." in prompt
        assert "Seg OK." in prompt
        assert "Seg ruim." in prompt
        assert "KEEP" in prompt

    def test_revision_prompt_includes_per_segment_verdicts(self):
        previous_segments = [
            SynthesizerSegment(text="Seg OK.", refs=[Reference(source="producao", type="sql")]),
            SynthesizerSegment(text="Seg ruim.", refs=[]),
        ]
        verifier_output = VerifierOutput(
            segments=[
                SegmentVerdict(index=0, verdict="supported", reasoning="Dados confirmados."),
                SegmentVerdict(index=1, verdict="not_supported", reasoning="Sem evidencia."),
            ],
            completeness=CompletenessCheck(covered=False, missing_aspects=["Faltou comparacao"]),
            overall_pass=False,
            feedback="Segmento 2 nao suportado.",
        )
        prompt = _build_revision_prompt("<evidence/>", previous_segments, "Feedback.", verifier_output)
        assert 'verdict="supported"' in prompt
        assert 'verdict="not_supported"' in prompt
        assert "<verifier_reasoning>Dados confirmados.</verifier_reasoning>" in prompt
        assert "<verifier_reasoning>Sem evidencia.</verifier_reasoning>" in prompt
        assert "Faltou comparacao" in prompt

    def test_revision_prompt_works_without_verifier_output(self):
        previous_segments = [SynthesizerSegment(text="Seg.", refs=[])]
        prompt = _build_revision_prompt("<evidence/>", previous_segments, "Feedback.")
        assert 'verdict="unknown"' in prompt
        assert "REVISION" in prompt


class TestSynthesizerNodeRetry:
    @patch("src.agent.nodes.synthesizer._invoke_synthesizer")
    @patch("src.agent.nodes.synthesizer.get_node_llm")
    def test_uses_initial_prompt_on_first_pass(self, mock_get_llm, mock_invoke):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_output = SynthesizerOutput(segments=[SynthesizerSegment(text="Answer.", refs=[])])
        mock_invoke.return_value = (mock_output, {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        state = {
            "question": "Test?",
            "sql_results": [],
            "text_results": [],
        }
        synthesizer_node(state)
        call_args = mock_invoke.call_args
        messages = call_args[0][1]
        system_content = messages[0].content
        assert "REVISION" not in system_content

    @patch("src.agent.nodes.synthesizer._invoke_synthesizer")
    @patch("src.agent.nodes.synthesizer.get_node_llm")
    def test_uses_revision_prompt_on_retry(self, mock_get_llm, mock_invoke):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_output = SynthesizerOutput(segments=[SynthesizerSegment(text="Revised.", refs=[])])
        mock_invoke.return_value = (mock_output, {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

        verifier_out = VerifierOutput(
            segments=[SegmentVerdict(index=0, verdict="not_supported", reasoning="Sem evidencia.")],
            completeness=CompletenessCheck(covered=True),
            overall_pass=False,
            feedback="Segmento 1 nao suportado.",
        )
        state = {
            "question": "Test?",
            "sql_results": [],
            "text_results": [],
            "verifier_output": verifier_out,
            "synthesizer_output": SynthesizerOutput(segments=[SynthesizerSegment(text="Old answer.", refs=[])]),
        }
        synthesizer_node(state)
        call_args = mock_invoke.call_args
        messages = call_args[0][1]
        system_content = messages[0].content
        assert "REVISION" in system_content
        assert "Segmento 1 nao suportado." in system_content
        assert "Old answer." in system_content
        assert 'verdict="not_supported"' in system_content
