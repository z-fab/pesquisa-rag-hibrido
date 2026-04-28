# src/agent/nodes/simple_synthesizer.py
import time

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.agent.state import AgentState
from src.config.providers import get_node_llm
from src.config.settings import SETTINGS
from src.services.evidence_service import build_evidence_context
from src.utils.tracking import extract_usage_from_response, get_text_content, record_end


def simple_synthesizer_node(state: AgentState) -> dict:
    """Simple synthesizer node: generates final answer directly from evidence (POC behavior)."""
    logger.info("Executing SimpleSynthesizer...")
    node_start = time.perf_counter()

    llm = get_node_llm("synthesis", SETTINGS)
    evidence = build_evidence_context(state)

    system_prompt = f"""<system>
<role>
    You are a synthesis agent responsible for answering the user's question
    using ONLY the structured and unstructured evidence provided below.
    Your task is to generate a clear, correct, and well-founded answer.
</role>

<constraints>
    - You MUST NOT infer, assume, or hallucinate information.
    - You MUST NOT use external knowledge.
    - You MUST use exclusively the data presented in the evidence sections.
    - All assertions must be supported by explicit evidence.
    - If information is not available in the evidence, say so explicitly.
</constraints>

<citation_rules>
    - For structured data, cite the table name from the query or data source.
    - For unstructured data, cite the PDF filename as listed in the context.
    - Recommended format: "According to table X..." or "According to report Y..."
</citation_rules>

<answer_style>
    - Be concise, objective, and direct.
    - Do not repeat the question.
    - Do not write superfluous text.
    - Do not generate methodological explanations.
    - Answer in clear, natural language.
</answer_style>

<conflict_resolution>
    - If structured and unstructured data diverge, report the divergence neutrally.
    - Do not attempt to reconcile by inventing values.
    - If data is insufficient, answer only the part that is possible and indicate what is missing.
    - If no relevant data exists, return: "There is insufficient evidence to answer."
</conflict_resolution>

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

    provider = SETTINGS.PROVIDER
    model_name = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
    tracking = record_end("simple_synthesizer", provider, model_name, node_start, usage)

    logger.debug(f"SimpleSynthesizer answer length: {len(answer)} chars")

    return {
        "final_answer": answer,
        **tracking,
    }
