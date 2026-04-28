# src/agent/nodes/simple_router.py
import time

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.agent.state import AgentState, PlannerOutput, SearchTask
from src.config.providers import get_node_llm
from src.config.settings import SETTINGS
from src.services.semantic_map_service import (
    format_structured_map_summary,
    format_unstructured_map_to_context,
)
from src.utils.tracking import extract_usage_from_response, parse_llm_json, record_end


def _build_router_prompt() -> str:
    """Builds the system prompt for the simple router."""
    return f"""<system>
<role>
    You are a QUERY ROUTER. Your task is to analyze the user's question and decide
    WHICH DATA SOURCE should be used.
    Classify the question as: structured, non_structured, or hybrid.
</role>

<data_sources>
    <structured>
        Use when the question requires ONLY quantitative values from SQL tables.
        Examples:
            - "Qual foi a producao de X?"
            - "Qual estado produziu mais?"
            - "Quantas toneladas...?"
    </structured>

    <non_structured>
        Use when the question requires ONLY qualitative/textual information from PDF reports.
        Examples:
            - "Quais sao os desafios...?"
            - "O que dizem os relatorios sobre...?"
            - "Quais problemas trabalhistas...?"
    </non_structured>

    <hybrid>
        Use when the answer requires combining:
            (1) numbers from SQL
            (2) qualitative explanations from reports
        Examples:
            - "Como X se compara a Y?"
            - "Por que X afeta Y?"
            - "X aparece em Y? O que isso indica?"
    </hybrid>
</data_sources>

<output_format>
    Return ONLY JSON, no extra explanations, in this format:
    {{
        "datasource": "structured | non_structured | hybrid",
        "tables": ["table1", ...],
        "documents": ["doc1.pdf", ...]
    }}
</output_format>

{format_structured_map_summary()}

{format_unstructured_map_to_context()}
</system>"""


def simple_router_node(state: AgentState) -> dict:
    """Simple router node: classifies question and produces PlannerOutput-compatible output."""
    logger.info("Executing SimpleRouter...")
    node_start = time.perf_counter()

    llm = get_node_llm("router", SETTINGS)
    question = state["question"]

    messages = [
        SystemMessage(content=_build_router_prompt()),
        HumanMessage(content=f"<user_query>{question}</user_query>"),
    ]

    response = llm.invoke(messages)
    usage = extract_usage_from_response(response)

    decision = parse_llm_json(response)
    if decision is None:
        logger.warning("Failed to parse router output, defaulting to structured")
        decision = {"datasource": "structured", "tables": [], "documents": []}

    # Coerção de shape — modelos pequenos às vezes embrulham em lista
    if isinstance(decision, list):
        # Lista com um dict dentro → desembrulha
        if len(decision) == 1 and isinstance(decision[0], dict):
            decision = decision[0]
        # Lista de múltiplas decisões → pega a primeira
        elif decision and isinstance(decision[0], dict):
            logger.warning(f"Router returned {len(decision)} decisions; using first")
            decision = decision[0]
        else:
            logger.warning("Router returned list without usable dict; defaulting to structured")
            decision = {"datasource": "structured", "tables": [], "documents": []}
    if not isinstance(decision, dict):
        logger.warning(f"Router returned non-dict ({type(decision).__name__}); defaulting to structured")
        decision = {"datasource": "structured", "tables": [], "documents": []}

    datasource = (decision.get("datasource") or "structured").strip().lower()
    tables = decision.get("tables") or []
    documents = decision.get("documents") or []

    # Convert to PlannerOutput (original question as query, no decomposition)
    searches: list[SearchTask] = []
    if datasource in ("structured", "hybrid"):
        searches.append(SearchTask(type="sql", query=question, sources=tables))
    if datasource in ("non_structured", "hybrid"):
        searches.append(SearchTask(type="text", query=question, sources=documents))

    provider = SETTINGS.PROVIDER
    model_name = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
    tracking = record_end("simple_router", provider, model_name, node_start, usage)

    logger.debug(f"SimpleRouter decision: {datasource}, searches: {len(searches)}")

    return {
        "needs_retrieval": True,
        "planner_output": PlannerOutput(searches=searches),
        **tracking,
    }
