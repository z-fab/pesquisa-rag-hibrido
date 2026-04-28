import time

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.agent.state import AgentState
from src.config.providers import get_node_llm
from src.config.settings import SETTINGS
from src.services.evidence_service import build_evidence_context
from src.utils.tracking import extract_usage_from_response, get_text_content, record_end


def consolidator_lite_node(state: AgentState) -> dict:
    """Consolidator lite: generates answer from raw evidence via LLM and adds references."""
    logger.info("Executing ConsolidatorLite...")
    node_start = time.perf_counter()

    llm = get_node_llm("synthesis", SETTINGS)
    evidence = build_evidence_context(state)

    system_prompt = f"""<system>
<role>
    You are a synthesis agent. Answer the user's question using ONLY the evidence provided below.
</role>

<constraints>
    - You MUST NOT hallucinate or fabricate information.
    - You MUST NOT use external knowledge.
    - Use exclusively the data in the evidence sections.
    - If information is not available, say so explicitly.
</constraints>

<answer_style>
    - Answer in clear, natural language.
    - Be complete: address all aspects of the question.
    - Cite your sources inline (e.g., "According to table X..." or "According to report Y...").
    - Do not repeat the question.
</answer_style>

<evidence>
{evidence}
</evidence>
</system>"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"<user_query>{state['question']}</user_query>"),
    ]

    response = llm.invoke(messages)
    usage = extract_usage_from_response(response)
    answer = get_text_content(response)

    # Build references section from sources
    ref_items: list[str] = []
    seen: set[str] = set()

    for res in state.get("sql_results", []):
        for source in res.get("sources", []):
            if source not in seen:
                seen.add(source)
                ref_items.append(f"[{len(ref_items) + 1}] {source} (SQL)")

    for res in state.get("text_results", []):
        for source in res.get("sources", []):
            if source not in seen:
                seen.add(source)
                ref_items.append(f"[{len(ref_items) + 1}] {source}")

    final_answer = f"{answer}\n\nReferências:\n" + "\n".join(ref_items) if ref_items else answer

    provider = SETTINGS.PROVIDER
    model_name = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
    tracking = record_end("consolidator_lite", provider, model_name, node_start, usage)

    logger.debug(f"ConsolidatorLite answer length: {len(final_answer)} chars, refs: {len(ref_items)}")

    return {
        "final_answer": final_answer,
        **tracking,
    }
