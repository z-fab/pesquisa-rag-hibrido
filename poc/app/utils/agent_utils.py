import time
from typing import Dict, Optional

from agent.state import AgentState


def extract_usage_from_response(response) -> Optional[Dict[str, float]]:
    """Extract token usage from LangChain response object.

    Handles both standardized usage_metadata (LangChain 0.2+) and
    OpenAI-specific response_metadata formats.
    """
    # Try standardized usage_metadata first (preferred)
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        metadata = response.usage_metadata
        # Already in standard format: input_tokens, output_tokens, total_tokens
        if isinstance(metadata, dict):
            return {
                "input_tokens": float(metadata.get("input_tokens", 0)),
                "output_tokens": float(metadata.get("output_tokens", 0)),
                "total_tokens": float(metadata.get("total_tokens", 0)),
            }

    # Try OpenAI-specific response_metadata
    if hasattr(response, "response_metadata") and response.response_metadata:
        token_data = response.response_metadata.get("token_usage", {})
        if token_data:
            # Map OpenAI format to standard format
            return {
                "input_tokens": float(token_data.get("prompt_tokens", 0)),
                "output_tokens": float(token_data.get("completion_tokens", 0)),
                "total_tokens": float(token_data.get("total_tokens", 0)),
            }

    # No usage found
    return None


def normalize_usage(raw: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not raw:
        return {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0}
    input_tokens = raw.get("input_tokens") or raw.get("prompt_tokens") or 0.0
    output_tokens = raw.get("output_tokens") or raw.get("completion_tokens") or 0.0
    total_tokens = (
        raw.get("total_tokens") or raw.get("total") or (input_tokens + output_tokens)
    )
    return {
        "input_tokens": input_tokens or 0.0,
        "output_tokens": output_tokens or 0.0,
        "total_tokens": total_tokens or 0.0,
    }


def record_end(
    state: AgentState, name: str, start_time: float, usage: Optional[Dict[str, float]]
) -> dict:
    """Records node execution metrics and returns state updates."""
    duration = round(time.perf_counter() - start_time, 4)

    # Build trace entry
    trace_entry = {"node": name, "duration": duration}

    # Build executed agents list (exclude router)
    executed = [name] if name != "router" else []

    # Normalize token usage (the custom reducer will sum it)
    norm = normalize_usage(usage)

    return {
        "trace": [trace_entry],  # Will be appended via operator.add
        "executed_agents": executed,  # Will be appended via operator.add
        "token_usage": norm,  # Will be summed via add_token_usage reducer
    }
