"""Tests for eval/experiments/verifier_signal.py — focados no parser AST."""

import pytest

from eval.experiments.verifier_signal import (
    _extract_signal_features,
    _parse_synthesizer_output_repr,
    _reconstruct_state,
)
from src.agent.state import (
    CompletenessCheck,
    Reference,
    SegmentVerdict,
    SynthesizerOutput,
    SynthesizerSegment,
    VerifierOutput,
)


class TestParseSynthesizerOutputRepr:
    def test_parse_simple_single_segment_no_refs(self):
        so_repr = "segments=[SynthesizerSegment(text='Olá mundo', refs=[])]"
        so = _parse_synthesizer_output_repr(so_repr)
        assert isinstance(so, SynthesizerOutput)
        assert len(so.segments) == 1
        assert so.segments[0].text == "Olá mundo"
        assert so.segments[0].refs == []

    def test_parse_segment_with_ref(self):
        so_repr = (
            "segments=[SynthesizerSegment(text='Soja produz bem', "
            "refs=[Reference(source='producao', type='sql', section='')])]"
        )
        so = _parse_synthesizer_output_repr(so_repr)
        assert len(so.segments) == 1
        assert len(so.segments[0].refs) == 1
        assert so.segments[0].refs[0].source == "producao"
        assert so.segments[0].refs[0].type == "sql"

    def test_parse_multiple_segments(self):
        so_repr = (
            "segments=["
            "SynthesizerSegment(text='Primeiro', refs=[Reference(source='a.pdf', type='text', section='s1')]), "
            "SynthesizerSegment(text='Segundo', refs=[])"
            "]"
        )
        so = _parse_synthesizer_output_repr(so_repr)
        assert len(so.segments) == 2
        assert so.segments[0].text == "Primeiro"
        assert so.segments[1].refs == []

    def test_parse_without_segments_prefix(self):
        # Se vier sem o prefixo `segments=`, deve funcionar também
        so_repr = "[SynthesizerSegment(text='Direto', refs=[])]"
        so = _parse_synthesizer_output_repr(so_repr)
        assert len(so.segments) == 1

    def test_parse_rejects_unknown_class(self):
        so_repr = "segments=[MaliciousClass(text='hack')]"
        with pytest.raises(ValueError, match="Classe não permitida"):
            _parse_synthesizer_output_repr(so_repr)

    def test_parse_rejects_invalid_syntax(self):
        so_repr = "segments=[SynthesizerSegment(text='unclosed"
        with pytest.raises(ValueError, match="Sintaxe inválida"):
            _parse_synthesizer_output_repr(so_repr)

    def test_parse_real_snapshot_format(self):
        # Formato real visto no snapshot no-verifier do GPT-5
        so_repr = (
            "segments=[SynthesizerSegment(text='Na safra 2025/26, os cinco estados "
            "brasileiros com maior produção de soja são Mato Grosso (MT), Paraná (PR).', "
            "refs=[Reference(source='conab_safras_soja', type='sql', section='')])]"
        )
        so = _parse_synthesizer_output_repr(so_repr)
        assert len(so.segments) == 1
        assert "Mato Grosso" in so.segments[0].text


class TestReconstructState:
    def test_reconstruct_from_string_synthesizer_output(self):
        item = {
            "input": {"question": "qual a produção de soja?"},
            "output": {
                "synthesizer_output": "segments=[SynthesizerSegment(text='texto', refs=[])]",
                "sql_results": [{"sql_query": "SELECT 1"}],
                "text_results": [],
            },
        }
        state = _reconstruct_state(item)
        assert state["question"] == "qual a produção de soja?"
        assert isinstance(state["synthesizer_output"], SynthesizerOutput)
        assert len(state["synthesizer_output"].segments) == 1
        assert state["sql_results"] == [{"sql_query": "SELECT 1"}]

    def test_reconstruct_from_dict_synthesizer_output(self):
        item = {
            "input": {"question": "q"},
            "output": {
                "synthesizer_output": {"segments": [{"text": "t", "refs": []}]},
                "sql_results": [],
                "text_results": [],
            },
        }
        state = _reconstruct_state(item)
        assert state["synthesizer_output"].segments[0].text == "t"


class TestExtractSignalFeatures:
    def test_extract_basic(self):
        vo = VerifierOutput(
            segments=[
                SegmentVerdict(index=0, verdict="supported", reasoning="ok"),
                SegmentVerdict(index=1, verdict="not_supported", reasoning="fail"),
                SegmentVerdict(index=2, verdict="partial", reasoning="meh"),
            ],
            completeness=CompletenessCheck(covered=False, missing_aspects=["a", "b"]),
            overall_pass=False,
            feedback="needs work",
        )
        feats = _extract_signal_features(vo, latency=5.123)
        assert feats["overall_pass"] is False
        assert feats["completeness_covered"] is False
        assert feats["n_missing_aspects"] == 2
        assert feats["n_segments"] == 3
        assert feats["n_supported"] == 1
        assert feats["n_not_supported"] == 1
        assert feats["n_partial"] == 1
        assert feats["pct_supported"] == pytest.approx(1 / 3)
        assert feats["verifier_latency"] == 5.123

    def test_extract_empty_segments(self):
        vo = VerifierOutput(
            segments=[],
            completeness=CompletenessCheck(covered=True, missing_aspects=[]),
            overall_pass=True,
            feedback="",
        )
        feats = _extract_signal_features(vo, latency=1.0)
        assert feats["n_segments"] == 0
        assert feats["pct_supported"] is None
