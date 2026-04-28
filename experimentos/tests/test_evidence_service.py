from src.services.evidence_service import build_evidence_context


class TestBuildEvidenceContext:
    def test_sql_evidence(self):
        state = {
            "sql_results": [
                {
                    "task_query": "Quais estados produzem mais soja?",
                    "sql_query": "SELECT estado FROM producao",
                    "result_raw": "[('MT', 3800), ('PR', 3500)]",
                    "sources": ["producao"],
                    "executed": True,
                    "error": "",
                },
            ],
            "text_results": [],
        }
        result = build_evidence_context(state)
        assert '<sql_evidence index="1">' in result
        assert "<task_query>Quais estados produzem mais soja?</task_query>" in result
        assert "<tables>producao</tables>" in result
        assert "<result>[('MT', 3800), ('PR', 3500)]</result>" in result

    def test_text_evidence(self):
        state = {
            "sql_results": [],
            "text_results": [
                {
                    "task_query": "O que e MIP?",
                    "chunks": [
                        {"content": "O Manejo Integrado...", "source": "manejo.pdf", "Header 1": "Capitulo 2"},
                    ],
                    "sources": ["manejo.pdf"],
                },
            ],
        }
        result = build_evidence_context(state)
        assert '<text_evidence index="1">' in result
        assert "<task_query>O que e MIP?</task_query>" in result
        assert "<content>O Manejo Integrado...</content>" in result
        assert "<source>manejo.pdf</source>" in result
        assert "<Header 1>Capitulo 2</Header 1>" in result

    def test_mixed_evidence(self):
        state = {
            "sql_results": [
                {
                    "task_query": "Q1",
                    "sql_query": "SELECT 1",
                    "result_raw": "1",
                    "sources": [],
                    "executed": True,
                    "error": "",
                },
            ],
            "text_results": [
                {"task_query": "Q2", "chunks": [{"content": "text", "source": "a.pdf"}], "sources": ["a.pdf"]},
            ],
        }
        result = build_evidence_context(state)
        assert '<sql_evidence index="1">' in result
        assert '<text_evidence index="1">' in result

    def test_empty_evidence(self):
        state = {"sql_results": [], "text_results": []}
        result = build_evidence_context(state)
        assert result == "No evidence available."

    def test_sql_evidence_includes_sql_query(self):
        state = {
            "sql_results": [
                {
                    "task_query": "Q",
                    "sql_query": "SELECT * FROM producao",
                    "result_raw": "[]",
                    "sources": [],
                    "executed": True,
                    "error": "",
                },
            ],
            "text_results": [],
        }
        result = build_evidence_context(state)
        assert "<sql_query>SELECT * FROM producao</sql_query>" in result
