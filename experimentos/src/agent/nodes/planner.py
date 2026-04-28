import time

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.agent.state import AgentState, PlannerOutput
from src.config.providers import get_node_llm, maybe_with_structured_output
from src.config.settings import SETTINGS
from src.services.semantic_map_service import (
    format_structured_map_summary,
    format_unstructured_map_to_context,
)
from src.utils.tracking import extract_usage_from_response, parse_llm_json, record_end


def _invoke_planner(llm, messages, schema):
    """Invoke planner LLM and return (PlannerOutput, usage).

    Detects structured output vs fallback by inspecting the response type:
    - dict with "parsed" key → structured output path
    - AIMessage → fallback JSON parsing path
    """
    result = llm.invoke(messages)

    if isinstance(result, dict) and "parsed" in result:
        if result["parsed"] is not None:
            return result["parsed"], extract_usage_from_response(result["raw"])
        # Structured output parsing failed; fall through to manual JSON parsing
        result = result["raw"]

    usage = extract_usage_from_response(result)
    parsed = parse_llm_json(result)
    if parsed:
        parsed = _coerce_planner_shape(parsed)
        return schema.model_validate(parsed), usage
    raise ValueError("Failed to parse planner output")


def _coerce_planner_shape(parsed):
    """Tolera desvios comuns de formato em modelos pequenos.

    Casos:
    - `{"type": "sql", "query": "...", "sources": [...]}` → {"searches": [ele]}
    - `[{"type": ...}, {...}]` → {"searches": [...]}
    - Já no formato {"searches": [...]} → passa direto.
    """
    if isinstance(parsed, dict):
        if "searches" in parsed:
            return parsed
        # Um objeto de search solto — embrulha
        if {"type", "query"}.issubset(parsed.keys()):
            return {"searches": [parsed]}
    if isinstance(parsed, list):
        # Lista de searches sem o wrapper
        if all(isinstance(it, dict) and "type" in it for it in parsed):
            return {"searches": parsed}
    return parsed


_OUTPUT_FORMAT = """<output_format>
    Return a JSON object with a "searches" array. Each item has:
    - "type": "sql" for structured/SQL data, "text" for unstructured/document data
    - "query": a natural language question (NOT SQL, NOT code) that will be sent to the next processing node
    - "sources": list of suggested table names or document IDs to search

    IMPORTANT: The "query" field must always be a question in natural language.
    For "sql" type, another node will generate the actual SQL query from your question.
    For "text" type, another node will perform a similarity search using your question.
    Never write SQL, code, or technical queries — only plain language questions.

    Example:
    {{
        "searches": [
            {{"type": "sql", "query": "Quais os cinco estados que mais produzem soja na safra 2025/26 e qual a produtividade de cada um?", "sources": ["producao"]}},
            {{"type": "text", "query": "Quais foram as principais metas estratégicas para 2024?", "sources": ["relatorio_2024.pdf"]}}
        ]
    }}

    Rules:
    - Only include search items for source types that are genuinely needed.
    - You may include multiple items of the same type if different sub-queries are needed.
</output_format>"""


def _build_initial_prompt() -> str:
    return f"""<system>
<role>
    You are a QUERY PLANNER. Your task is to analyze the user's question and decompose it
    into one or more sub-queries, each targeting a specific data source.
</role>

<data_sources>
    <sql>
        Use when a sub-query requires quantitative values from SQL tables.
    </sql>
    <text>
        Use when a sub-query requires qualitative/textual information from PDF reports.
    </text>
</data_sources>

<instructions>
    - Analyze the user's question and determine which data sources are needed.
    - Reformulate specialized sub-queries for each source type.
    - Each sub-query should be self-contained and specific to its source type.
    - Suggest relevant tables or documents for each sub-query based on the available schema.
</instructions>

<routing_guidelines>
    Use SQL ONLY when the question can be fully answered with quantitative data
    (numbers, rankings, statistics, temporal evolution, aggregations).

    Use TEXT ONLY when the question asks about concepts, definitions, processes,
    techniques, recommendations, or qualitative descriptions.

    Use BOTH when the question explicitly requires quantitative data AND
    qualitative context that cannot be derived from numbers alone.
    Also use BOTH when a complete answer requires CONNECTING quantitative data
    with qualitative context — even if the question reads as a single question,
    if understanding the "why" or "how" behind the numbers requires document
    knowledge, route to both sources.

    Examples — SQL only (do NOT add text):
    - "Quais os 5 maiores produtores de soja?" -> numbers/ranking
    - "Como evoluiu a producao ao longo das decadas?" -> temporal aggregation
    - "Qual a area colhida que superou X hectares?" -> filtering/comparison

    Examples — TEXT only:
    - "O que e o manejo integrado de pragas?" -> definition/concept
    - "Quais as praticas recomendadas para plantio direto?" -> recommendations

    Examples — BOTH (hybrid):
    - "Qual a producao de cafe em Rondonia e quais variedades sao cultivadas?"
      -> numbers (SQL) + varieties/descriptions (TEXT)
    - "Quais os maiores produtores de soja e quais desafios fitossanitarios enfrentam?"
      -> ranking (SQL) + qualitative challenges (TEXT)
    - "Como a adocao do plantio direto impactou a produtividade da soja?"
      -> productivity trends (SQL) + what SPD is and how it works (TEXT)
      The answer requires RELATING quantitative evolution with qualitative context.
    - "Quais culturas cresceram mais no Cerrado e por que essa regiao se tornou tao relevante?"
      -> production growth data (SQL) + geographic/agronomic factors (TEXT)
      A complete answer needs to CONNECT the numbers with the explanations.
</routing_guidelines>

{_OUTPUT_FORMAT}

{format_structured_map_summary()}

{format_unstructured_map_to_context()}
</system>"""


def _build_refinement_prompt(feedback: str) -> str:
    return f"""<system>
<role>
    You are a QUERY PLANNER performing a refinement pass. A previous answer was reviewed
    and found to have gaps. Your task is to generate additional sub-queries to fill those gaps.
</role>

<data_sources>
    <sql>
        Use when a sub-query requires quantitative values from SQL tables.
    </sql>
    <text>
        Use when a sub-query requires qualitative/textual information from PDF reports.
    </text>
</data_sources>

<instructions>
    - The original user question is provided for context.
    - Focus on the verifier's feedback to understand what information is missing.
    - Generate ONLY the sub-queries needed to fill the identified gaps.
    - Do NOT repeat sub-queries that were already answered successfully.
    - Only include sub-queries for sources that will genuinely help address the gaps.
</instructions>

<verifier_feedback>
    {feedback}
</verifier_feedback>

{_OUTPUT_FORMAT}

{format_structured_map_summary()}

{format_unstructured_map_to_context()}
</system>"""


def planner_node(state: AgentState) -> dict:
    """Planner node: decomposes the question into typed sub-queries."""
    logger.info("Executing Planner...")

    node_start = time.perf_counter()

    base_llm = get_node_llm("planner", SETTINGS)
    llm = maybe_with_structured_output(base_llm, PlannerOutput)

    verifier_out = state.get("verifier_output")
    feedback = verifier_out.feedback if verifier_out else None
    system_prompt = _build_refinement_prompt(feedback) if feedback else _build_initial_prompt()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"<user_query>{state['question']}</user_query>"),
    ]

    planner_output, usage = _invoke_planner(llm, messages, PlannerOutput)

    provider = SETTINGS.PROVIDER
    model_name = getattr(base_llm, "model_name", getattr(base_llm, "model", "unknown"))
    tracking = record_end("planner", provider, model_name, node_start, usage)

    logger.debug(f"Planner output: {planner_output}")

    return {
        "needs_retrieval": True,
        "planner_output": planner_output,
        **tracking,
    }
