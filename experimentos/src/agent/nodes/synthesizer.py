import time

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.agent.state import AgentState, SynthesizerOutput
from src.config.providers import get_node_llm, maybe_with_structured_output
from src.config.settings import SETTINGS
from src.services.evidence_service import build_evidence_context
from src.utils.tracking import extract_usage_from_response, parse_llm_json, record_end

_SYNTHESIS_RULES = """<constraints>
    - You MUST NOT hallucinate or fabricate information.
    - You MAY derive comparisons, trends, and rankings directly from the evidence data.
    - You MUST NOT speculate about causes unless the evidence explicitly states them.
    - You MUST NOT use external knowledge.
    - You MUST use exclusively the data presented in the evidence sections.
    - All factual assertions must be supported by explicit evidence and referenced.
    - If information is not available in the evidence, say so explicitly.
</constraints>

<answer_style>
    - Answer in clear, natural language. Do not repeat the question.
    - Be thorough: address ALL aspects of the original question.
    - When presenting numerical data, add comparative analysis:
      highlight rankings, notable differences, outliers, and proportions
      (e.g., "X produces more than the next four combined").
    - When data shows temporal evolution, describe the trend
      (growth, decline, stability) and notable inflection points.
    - When multiple entities are compared, highlight what distinguishes
      them (highest, lowest, fastest growth, etc.).
    - These analyses are NOT hallucination — they are interpretations
      grounded in the evidence. Only state what the data directly shows.
    - Structure the answer with one segment per major aspect of the question.
</answer_style>

<conflict_resolution>
    - If structured and unstructured data diverge, report the divergence neutrally.
    - Do not attempt to reconcile by inventing values.
    - If data is insufficient, answer only the part that is possible and indicate what is missing.
    - If no relevant data exists, return a single segment stating insufficient evidence.
</conflict_resolution>"""

_OUTPUT_FORMAT = """<output_format>
    Return a JSON object with a "segments" array. Each segment has:
    - "text": a paragraph or sentence of the answer in natural language
    - "refs": a list of references supporting this text segment (can be empty)

    Each reference has:
    - "source": the exact filename (e.g. "relatorio.pdf") or table name (e.g. "producao")
    - "type": "sql" for structured data sources, "text" for document sources
    - "section": the section/chapter header if available (only for "text" type, empty string otherwise)

    Rules:
    - Every factual claim MUST have at least one reference.
    - Transitional or concluding text without specific data can have empty refs.
    - Use ONLY sources that appear in the evidence. Never invent sources.
    - The "source" field must exactly match a source name from the evidence.
    - For text sources, populate "section" with the most specific header available in the chunk metadata.

    Example:
    {{
        "segments": [
            {{
                "text": "A soja e a principal cultura do Mato Grosso, com produtividade media de 3.800 kg/ha.",
                "refs": [{{"source": "producao", "type": "sql", "section": ""}}]
            }},
            {{
                "text": "O Manejo Integrado de Pragas define niveis de acao baseados em amostragem periodica.",
                "refs": [{{"source": "manejo.pdf", "type": "text", "section": "Secao 2.1"}}]
            }},
            {{
                "text": "Essas praticas contribuem para a sustentabilidade da producao.",
                "refs": []
            }}
        ]
    }}
</output_format>"""


def _invoke_synthesizer(llm, messages, schema):
    """Invoke synthesizer LLM and return (SynthesizerOutput, usage).

    Detects structured output vs fallback by inspecting the response type:
    - dict with "parsed" key -> structured output path
    - AIMessage -> fallback JSON parsing path
    """
    result = llm.invoke(messages)

    if isinstance(result, dict) and "parsed" in result:
        if result["parsed"] is not None:
            return result["parsed"], extract_usage_from_response(result["raw"])
        result = result["raw"]

    usage = extract_usage_from_response(result)
    parsed = parse_llm_json(result)
    if parsed:
        parsed = _coerce_synthesizer_shape(parsed)
        return schema.model_validate(parsed), usage
    raise ValueError("Failed to parse synthesizer output")


def _coerce_synthesizer_shape(parsed):
    """Normaliza desvios comuns em modelos pequenos antes de validar SynthesizerOutput.

    Caso principal: lista de segmentos solta (sem wrapper `{"segments": [...]}`).
    Outros casos possíveis: um único segmento dict solto.
    """
    if isinstance(parsed, list):
        # Lista direta de segmentos → embrulha
        if all(isinstance(it, dict) and "text" in it for it in parsed):
            return {"segments": parsed}
    if isinstance(parsed, dict):
        if "segments" in parsed:
            return parsed
        # Um segmento solto (tem 'text') → embrulha em lista
        if "text" in parsed:
            return {"segments": [parsed]}
    return parsed


def _build_initial_prompt(evidence: str) -> str:
    """Builds the system prompt for the first synthesis pass."""
    return f"""<system>
<role>
    You are a synthesis agent responsible for answering the user's question
    using ONLY the structured and unstructured evidence provided below.
</role>

{_SYNTHESIS_RULES}

{_OUTPUT_FORMAT}

<evidence>
{evidence}
</evidence>
</system>"""


def _build_revision_prompt(evidence: str, previous_segments: list, feedback: str, verifier_output=None) -> str:
    """Builds the system prompt for a REVISION pass after verifier feedback."""
    verdicts = {}
    if verifier_output:
        for sv in verifier_output.segments:
            verdicts[sv.index] = sv

    segments_xml_lines = []
    for i, seg in enumerate(previous_segments):
        verdict_info = verdicts.get(i)
        verdict_str = verdict_info.verdict if verdict_info else "unknown"
        segments_xml_lines.append(f'  <segment index="{i}" verdict="{verdict_str}">')
        segments_xml_lines.append(f"    <text>{seg.text}</text>")
        if seg.refs:
            segments_xml_lines.append("    <refs>")
            for r in seg.refs:
                segments_xml_lines.append(f'      <ref source="{r.source}" type="{r.type}" section="{r.section}" />')
            segments_xml_lines.append("    </refs>")
        if verdict_info and verdict_info.reasoning:
            segments_xml_lines.append(f"    <verifier_reasoning>{verdict_info.reasoning}</verifier_reasoning>")
        segments_xml_lines.append("  </segment>")
    segments_xml = "\n".join(segments_xml_lines)

    completeness_xml = ""
    if verifier_output and not verifier_output.completeness.covered:
        missing = verifier_output.completeness.missing_aspects
        if missing:
            completeness_xml = "\n<missing_aspects>\n" + "\n".join(f"  - {a}" for a in missing) + "\n</missing_aspects>"

    return f"""<system>
<role>
    You are a synthesis agent performing a REVISION pass. A previous answer was reviewed
    by a verifier and found to have issues. Your task is to revise the answer.
</role>

{_SYNTHESIS_RULES}

<revision_instructions>
    - Review the previous segments with their verdicts below.
    - KEEP segments with verdict="supported" exactly as they are — do not rewrite them.
    - REVISE segments with verdict="not_supported" or verdict="partial", using the evidence.
      Read the verifier_reasoning to understand what specifically needs fixing.
    - If missing_aspects are listed, ADD new segments to cover them.
    - You MUST still follow all constraints above (no hallucination, references required, etc).
</revision_instructions>

<previous_segments>
{segments_xml}
</previous_segments>

{completeness_xml}

<verifier_feedback>
{feedback}
</verifier_feedback>

{_OUTPUT_FORMAT}

<evidence>
{evidence}
</evidence>
</system>"""


def synthesizer_node(state: AgentState) -> dict:
    """Synthesizer node: combines all evidence into structured segments with references."""
    logger.info("Executing Synthesizer...")

    node_start = time.perf_counter()

    base_llm = get_node_llm("synthesis", SETTINGS)
    llm = maybe_with_structured_output(base_llm, SynthesizerOutput)

    evidence = build_evidence_context(state)
    logger.debug(f"Evidence context built: {len(evidence)} chars")

    verifier_output = state.get("verifier_output")
    previous_output = state.get("synthesizer_output")
    if verifier_output and previous_output:
        system_prompt = _build_revision_prompt(
            evidence, previous_output.segments, verifier_output.feedback, verifier_output
        )
        logger.debug(f"Using revision prompt with feedback: {verifier_output.feedback[:200]}")
    else:
        system_prompt = _build_initial_prompt(evidence)

    user_msg = f"<user_query>{state['question']}</user_query>"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg),
    ]

    output, usage = _invoke_synthesizer(llm, messages, SynthesizerOutput)

    logger.debug(
        f"Synthesizer returned {len(output.segments)} segments "
        f"with {sum(len(s.refs) for s in output.segments)} total refs"
    )

    provider = SETTINGS.PROVIDER
    model_name = getattr(base_llm, "model_name", getattr(base_llm, "model", "unknown"))
    tracking = record_end("synthesizer", provider, model_name, node_start, usage)

    return {
        "synthesizer_output": output,
        **tracking,
    }
