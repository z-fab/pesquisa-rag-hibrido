from langchain_core.messages import AIMessage

from src.utils.tracking import parse_llm_json


def _msg(content):
    return AIMessage(content=content)


class TestParseLLMJson:
    def test_plain_json(self):
        result = parse_llm_json(_msg('{"a": 1}'))
        assert result == {"a": 1}

    def test_markdown_fence_json(self):
        raw = '```json\n{"a": 1, "b": [2, 3]}\n```'
        assert parse_llm_json(_msg(raw)) == {"a": 1, "b": [2, 3]}

    def test_markdown_fence_no_lang(self):
        raw = '```\n{"a": 1}\n```'
        assert parse_llm_json(_msg(raw)) == {"a": 1}

    def test_xml_wrapper_output(self):
        raw = '<output>\n{"searches": [{"type": "sql"}]}\n</output>'
        assert parse_llm_json(_msg(raw)) == {"searches": [{"type": "sql"}]}

    def test_xml_wrapper_with_trailing_prose(self):
        """Model returns JSON in <output> tags then adds explanation text."""
        raw = (
            "<output>\n"
            '{"searches": [{"type": "sql", "query": "test", "sources": ["t"]}]}\n'
            "</output>\n\n"
            "O sistema identificou que a pergunta requer dados quantitativos..."
        )
        result = parse_llm_json(_msg(raw))
        assert result == {"searches": [{"type": "sql", "query": "test", "sources": ["t"]}]}

    def test_json_embedded_in_prose(self):
        raw = 'Here is my answer:\n\n{"verdict": "pass"}\n\nHope this helps!'
        assert parse_llm_json(_msg(raw)) == {"verdict": "pass"}

    def test_json_array(self):
        raw = '[{"a": 1}, {"a": 2}]'
        assert parse_llm_json(_msg(raw)) == [{"a": 1}, {"a": 2}]

    def test_gemini_content_blocks(self):
        """Gemini returns content as list of blocks, not string."""
        msg = AIMessage(content=[{"type": "text", "text": '{"a": 1}'}])
        assert parse_llm_json(msg) == {"a": 1}

    def test_json_with_nested_strings_containing_braces(self):
        raw = '{"msg": "use {curly} braces to denote sets"}'
        assert parse_llm_json(_msg(raw)) == {"msg": "use {curly} braces to denote sets"}

    def test_returns_none_on_unparseable(self):
        assert parse_llm_json(_msg("I don't know how to answer that")) is None

    def test_fallback_when_first_brace_is_invalid(self):
        """If the first '{' starts something non-JSON, scanner should try subsequent opens."""
        raw = 'Some text with { incomplete. Then real JSON: {"a": 1}'
        assert parse_llm_json(_msg(raw)) == {"a": 1}
