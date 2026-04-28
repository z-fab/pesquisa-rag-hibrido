import operator
from typing import Annotated, Dict, List, Optional, TypedDict


# Custom reducer for token_usage that sums the values
def add_token_usage(
    left: Optional[Dict[str, float]], right: Optional[Dict[str, float]]
) -> Dict[str, float]:
    """Custom reducer to sum token usage dictionaries."""
    if not left:
        left = {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0}
    if not right:
        right = {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0}
    return {
        "input_tokens": left.get("input_tokens", 0.0) + right.get("input_tokens", 0.0),
        "output_tokens": left.get("output_tokens", 0.0)
        + right.get("output_tokens", 0.0),
        "total_tokens": left.get("total_tokens", 0.0) + right.get("total_tokens", 0.0),
    }


class AgentState(TypedDict, total=False):
    question: str
    router_decision: str
    router_tables: List[str]
    router_docs: List[str]

    sql_query: str
    sql_executed: bool
    sql_result: List[dict]
    sql_result_raw: str

    text_result: str
    final_answer: str
    sources: List[str]
    trace: Annotated[List[dict], operator.add]
    executed_agents: Annotated[List[str], operator.add]
    total_latency: float
    token_usage: Annotated[Dict[str, float], add_token_usage]
    total_start: float
