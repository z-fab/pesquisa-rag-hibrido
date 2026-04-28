from src.agent.nodes.consolidator import consolidator_node
from src.agent.state import Reference, SynthesizerOutput, SynthesizerSegment


def _make_state(*segments_data):
    segments = []
    for text, refs in segments_data:
        ref_objs = [Reference(source=r[0], type=r[1], section=r[2] if len(r) > 2 else "") for r in refs]
        segments.append(SynthesizerSegment(text=text, refs=ref_objs))
    return {"synthesizer_output": SynthesizerOutput(segments=segments)}


class TestConsolidatorNode:
    def test_formats_segments_with_numbered_refs(self):
        state = _make_state(
            ("A soja e importante.", [("producao", "sql")]),
            ("O MIP define niveis de acao.", [("manejo.pdf", "text", "Secao 2.1")]),
        )

        result = consolidator_node(state)

        assert "A soja e importante. [1]" in result["final_answer"]
        assert "O MIP define niveis de acao. [2]" in result["final_answer"]
        assert "[1] producao (SQL)" in result["final_answer"]
        assert "[2] manejo.pdf \u2014 Secao 2.1" in result["final_answer"]

    def test_deduplicates_refs(self):
        state = _make_state(
            ("Primeiro trecho.", [("producao", "sql")]),
            ("Segundo trecho.", [("producao", "sql")]),
        )

        result = consolidator_node(state)

        assert "Primeiro trecho. [1]" in result["final_answer"]
        assert "Segundo trecho. [1]" in result["final_answer"]
        assert result["final_answer"].count("[1] producao (SQL)") == 1

    def test_segments_without_refs_no_markers(self):
        state = _make_state(
            ("Trecho com dados.", [("producao", "sql")]),
            ("Conclusao sem dados.", []),
        )

        result = consolidator_node(state)

        assert "Trecho com dados. [1]" in result["final_answer"]
        assert "Conclusao sem dados." in result["final_answer"]
        assert "Conclusao sem dados. [" not in result["final_answer"]

    def test_sql_ref_format(self):
        state = _make_state(("Dados.", [("producao", "sql")]))

        result = consolidator_node(state)

        assert "[1] producao (SQL)" in result["final_answer"]

    def test_text_ref_with_section(self):
        state = _make_state(("Dados.", [("relatorio.pdf", "text", "Capitulo 3")]))

        result = consolidator_node(state)

        assert "[1] relatorio.pdf \u2014 Capitulo 3" in result["final_answer"]

    def test_text_ref_without_section(self):
        state = _make_state(("Dados.", [("relatorio.pdf", "text")]))

        result = consolidator_node(state)

        assert "[1] relatorio.pdf" in result["final_answer"]
        assert "\u2014" not in result["final_answer"]

    def test_multiple_refs_per_segment(self):
        state = _make_state(
            ("Dados cruzados.", [("producao", "sql"), ("relatorio.pdf", "text", "Cap 1")]),
        )

        result = consolidator_node(state)

        assert "Dados cruzados. [1][2]" in result["final_answer"]

    def test_empty_segments(self):
        state = {}

        result = consolidator_node(state)

        assert result["final_answer"] == ""

    def test_reference_block_at_end(self):
        state = _make_state(("Trecho.", [("a.pdf", "text")]))

        result = consolidator_node(state)

        lines = result["final_answer"].strip().split("\n")
        assert lines[-1].startswith("[1]")
        assert any("Referências:" in line for line in lines)

    def test_executed_agents(self):
        state = {}

        result = consolidator_node(state)

        assert result["executed_agents"] == ["consolidator"]
