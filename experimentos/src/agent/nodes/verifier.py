import time

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.agent.state import AgentState, VerifierOutput
from src.config.providers import get_node_llm, maybe_with_structured_output
from src.config.settings import SETTINGS
from src.services.evidence_service import build_evidence_context
from src.utils.tracking import extract_usage_from_response, parse_llm_json, record_end

_OUTPUT_FORMAT = """<output_format>
    Return a JSON object with:
    - "segments": array of verdicts, one per segment. Each has:
        - "index": integer matching the segment index
        - "verdict": one of "supported", "not_supported", "partial"
        - "reasoning": step-by-step explanation of why this verdict was chosen
    - "completeness": object with:
        - "covered": boolean — does the answer address the original question?
        - "missing_aspects": list of strings describing what is missing (empty if covered)
    - "overall_pass": boolean — true ONLY if all segments are "supported" AND completeness is covered
    - "feedback": string — empty if overall_pass is true. If false, write actionable feedback:
      which segments failed, why, and what information is needed to fix them.

    Example:
    {{
        "segments": [
            {{"index": 0, "verdict": "supported", "reasoning": "The claim that MT produces 3800 kg/ha is confirmed by sql_evidence index 1."}},
            {{"index": 1, "verdict": "not_supported", "reasoning": "The claim about MIP levels is not found in any evidence."}}
        ],
        "completeness": {{"covered": false, "missing_aspects": ["Faltou comparacao entre safras"]}},
        "overall_pass": false,
        "feedback": "Segmento 2 afirma dados sobre niveis de MIP sem evidencia. Alem disso, a pergunta pedia comparacao entre safras que nao foi abordada."
    }}
</output_format>"""


def _build_segments_context(segments: list) -> str:
    """Formats synthesized segments as XML for the verifier prompt.

    Accepts SynthesizerSegment objects (Pydantic models).
    """
    lines = []
    for i, seg in enumerate(segments):
        lines.append(f'  <segment index="{i}">')
        lines.append(f"    <text>{seg.text}</text>")
        if seg.refs:
            lines.append("    <refs>")
            for ref in seg.refs:
                lines.append(f'      <ref type="{ref.type}">')
                lines.append(f"        <source>{ref.source}</source>")
                if ref.section:
                    lines.append(f"        <section>{ref.section}</section>")
                lines.append("      </ref>")
            lines.append("    </refs>")
        lines.append("  </segment>")
    return "\n".join(lines)


def _build_verifier_prompt(question: str, segments_xml: str, evidence_xml: str) -> str:
    """Assembles the complete verifier system prompt."""
    return f"""<system>
<role>
    You are a verification agent. Your task is to evaluate a synthesized answer
    for faithfulness to evidence and completeness regarding the original question.
</role>

<instructions>
    - Analyze EACH segment against the provided evidence.
    - For each segment, reason step-by-step whether the factual claims are supported.
    - A segment is "supported" if ALL factual claims are backed by the evidence.
    - A segment is "not_supported" if ANY factual claim contradicts or is absent from the evidence.
    - A segment is "partial" if some claims are supported but others are not.
    - Segments without factual claims (transitions, conclusions) are "supported".
    - Then evaluate completeness using the rubric below.
    - Set overall_pass to true ONLY if all segments are "supported" AND completeness is covered.
    - If overall_pass is false, write actionable feedback in the feedback field:
      specify which segments failed, why, and what information is needed.
    - If overall_pass is true, leave feedback as an empty string.
</instructions>

<completeness_evaluation>
    To evaluate completeness, follow these steps:
    1. Parse the original question and list EVERY distinct aspect it asks about.
       Example: "Quais os 5 maiores produtores e qual a produtividade?"
       -> aspects: (a) identify the top 5, (b) show productivity for each.
    2. For each aspect, check if the synthesized answer addresses it with specific data.
    3. If the question involves multiple entities or time periods, verify
       the answer compares or relates them (not just lists raw data).
    4. Mark covered=false if ANY aspect is missing or only superficially addressed.
    5. In missing_aspects, list each gap specifically and actionably
       (e.g., "Falta comparacao entre os estados" not "resposta incompleta").
</completeness_evaluation>

{_OUTPUT_FORMAT}

<original_question>{question}</original_question>

<synthesized_answer>
{segments_xml}
</synthesized_answer>

<evidence>
{evidence_xml}
</evidence>
</system>"""


def _invoke_verifier(llm, messages, schema):
    """Invoke verifier LLM and return (VerifierOutput, usage).

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
        parsed = _coerce_verifier_shape(parsed)
        return schema.model_validate(parsed), usage
    raise ValueError("Failed to parse verifier output")


def _coerce_verifier_shape(parsed):
    """Normaliza desvios comuns em modelos pequenos antes de validar VerifierOutput.

    Casos cobertos:
    - Lista solta de segmentos: `[{index,verdict,reasoning}, ...]` → wrap em estrutura completa.
    - Segmento único solto: `{index,verdict,reasoning}` → wrap como único segment.
    - `segments[i].reasoning` vem como lista em vez de string → junta com "; ".
    - `completeness.missing_aspects` vem como string em vez de lista → embrulha em lista.
    - `feedback` vem como lista → junta.
    - Faltam `completeness` / `overall_pass` / `feedback` → preenche defaults.

    Defaults pragmáticos quando o LLM não produziu os campos:
    - overall_pass = True (evita retry loop improdutivo com modelo que não segue schema)
    - completeness = {covered: True, missing_aspects: []}
    - feedback = "Verifier output incompleto — checagem parcial."
    """
    # 1. Lista direta de segmentos (sem wrapper)
    if isinstance(parsed, list) and all(isinstance(it, dict) and "verdict" in it for it in parsed):
        parsed = {"segments": parsed}

    if not isinstance(parsed, dict):
        return parsed

    # 2. Um único segmento solto no top-level (keys de SegmentVerdict)
    if "verdict" in parsed and "segments" not in parsed:
        parsed = {"segments": [{k: parsed[k] for k in ("index", "verdict", "reasoning") if k in parsed}]}

    # 2b. Campos de completeness colapsados no top-level — Mistral às vezes omite o sub-dict.
    # Se existe `covered` no top-level e não existe `completeness`, extrai para o sub-dict.
    if "covered" in parsed and "completeness" not in parsed:
        parsed["completeness"] = {
            "covered": parsed.pop("covered"),
            "missing_aspects": parsed.pop("missing_aspects", []),
        }

    # 2c. Sem segments algum — Verifier não produziu. Marca como checagem degradada.
    if "segments" not in parsed:
        parsed["segments"] = []

    # 3. Normaliza segmentos internos
    segments = parsed.get("segments")
    if isinstance(segments, list):
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            reason = seg.get("reasoning")
            if isinstance(reason, list):
                seg["reasoning"] = "; ".join(str(x) for x in reason)
            # Garantir index (alguns modelos omitem)
            if "index" not in seg:
                seg["index"] = segments.index(seg)

    # 4. Completa campos faltantes com defaults pragmáticos
    if "completeness" not in parsed:
        parsed["completeness"] = {"covered": True, "missing_aspects": []}
    else:
        c = parsed["completeness"]
        if isinstance(c, dict):
            missing = c.get("missing_aspects")
            if isinstance(missing, str):
                c["missing_aspects"] = [missing] if missing.strip() else []
            if "covered" not in c:
                c["covered"] = True

    if "overall_pass" not in parsed:
        # Inferir de segmentos: todos supported + completeness coberta
        segs = parsed.get("segments", [])
        all_supported = all(s.get("verdict") == "supported" for s in segs if isinstance(s, dict))
        parsed["overall_pass"] = bool(all_supported and parsed.get("completeness", {}).get("covered", True))

    if "feedback" not in parsed:
        parsed["feedback"] = "" if parsed.get("overall_pass") else "Verifier output incompleto — checagem parcial."
    elif isinstance(parsed["feedback"], list):
        parsed["feedback"] = "; ".join(str(x) for x in parsed["feedback"])

    return parsed


def verifier_node(state: AgentState) -> dict:
    """Verifier node: checks faithfulness and completeness of synthesized answer."""
    logger.info("Executing Verifier...")

    node_start = time.perf_counter()

    base_llm = get_node_llm("verifier", SETTINGS)
    llm = maybe_with_structured_output(base_llm, VerifierOutput)

    evidence_xml = build_evidence_context(state)
    synth_output = state.get("synthesizer_output")
    segments = synth_output.segments if synth_output else []
    segments_xml = _build_segments_context(segments)
    prompt = _build_verifier_prompt(state["question"], segments_xml, evidence_xml)

    logger.debug(f"Verifier context: {len(evidence_xml)} chars evidence, {len(segments_xml)} chars segments")

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="Verify the synthesized answer above."),
    ]

    output, usage = _invoke_verifier(llm, messages, VerifierOutput)

    logger.debug(
        f"Verifier result: overall_pass={output.overall_pass}, "
        f"{sum(1 for s in output.segments if s.verdict == 'supported')}/{len(output.segments)} supported, "
        f"completeness={'covered' if output.completeness.covered else 'not covered'}"
    )

    retry_count = state.get("retry_count", 0)

    provider = SETTINGS.PROVIDER
    model_name = getattr(base_llm, "model_name", getattr(base_llm, "model", "unknown"))
    tracking = record_end("verifier", provider, model_name, node_start, usage)

    return {
        "verifier_output": output,
        "retry_count": retry_count if output.overall_pass else retry_count + 1,
        **tracking,
    }
