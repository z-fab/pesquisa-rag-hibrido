import json
import re
import time

from loguru import logger

_XML_WRAPPER_RE = re.compile(r"^<(\w+)>\s*(.*?)\s*</\1>\s*", re.DOTALL)


def get_text_content(response) -> str:
    """Extracts text content from an LLM response, normalizing across providers.

    OpenAI returns response.content as a string.
    Gemini returns response.content as a list of content blocks:
        [{"type": "text", "text": "...", "extras": {...}}]

    This function always returns a plain string.
    """
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def _extract_first_json(text: str) -> dict | list | None:
    """Scan text for the first balanced { ... } or [ ... ] block that parses as JSON.

    Tolerates leading explanatory prose, trailing commentary, or the JSON
    appearing in the middle of the response. Respects brackets inside strings.
    """
    open_chars = {"{": "}", "[": "]"}
    for i, c in enumerate(text):
        if c not in open_chars:
            continue
        close = open_chars[c]
        depth = 0
        in_string = False
        escape = False
        for j in range(i, len(text)):
            ch = text[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == c:
                depth += 1
            elif ch == close:
                depth -= 1
                if depth == 0:
                    candidate = text[i : j + 1]
                    try:
                        # strict=False permite caracteres de controle (\n, \t, \r) dentro de strings
                        # — alguns LLMs (DeepSeek V3.2 notadamente) emitem newlines literais no JSON.
                        return json.loads(candidate, strict=False)
                    except (json.JSONDecodeError, ValueError):
                        break  # try the next opening bracket
    return None


def parse_llm_json(response) -> dict | None:
    """Extracts and parses JSON from an LLM response.

    Handles (in order):
    - plain JSON
    - markdown fences (```json ... ``` or ``` ... ```)
    - XML-like wrappers (<output>...</output>, <json>...</json>, etc.)
    - JSON embedded in prose (falls back to first balanced { ... } / [ ... ])

    Returns None and logs a warning if no JSON-like block can be parsed.
    """
    text = get_text_content(response)
    cleaned = text.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines[1:] if line.strip() != "```" and not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    # Strip outer XML-like wrapper (<output>...</output>, <json>...</json>, etc.)
    xml_match = _XML_WRAPPER_RE.match(cleaned)
    if xml_match:
        cleaned = xml_match.group(2).strip()

    # Direct parse (strict=False tolera \n, \t, \r literais dentro de strings)
    try:
        return json.loads(cleaned, strict=False)
    except (json.JSONDecodeError, ValueError):
        pass

    # Last resort: find first balanced JSON block anywhere in the text
    extracted = _extract_first_json(cleaned)
    if extracted is not None:
        return extracted

    logger.warning(f"Failed to parse LLM JSON.\nRaw: {text[:500]}")
    return None


def extract_usage_from_response(response) -> dict[str, float] | None:
    """Extract token usage from a LangChain LLM response.

    Handles both standardized usage_metadata and OpenAI-specific response_metadata.
    """
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        metadata = response.usage_metadata
        if isinstance(metadata, dict):
            return {
                "input_tokens": float(metadata.get("input_tokens", 0)),
                "output_tokens": float(metadata.get("output_tokens", 0)),
                "total_tokens": float(metadata.get("total_tokens", 0)),
            }

    if hasattr(response, "response_metadata") and response.response_metadata:
        token_data = response.response_metadata.get("token_usage", {})
        if token_data:
            return {
                "input_tokens": float(token_data.get("prompt_tokens", 0)),
                "output_tokens": float(token_data.get("completion_tokens", 0)),
                "total_tokens": float(token_data.get("total_tokens", 0)),
            }

    return None


def normalize_usage(raw: dict[str, float] | None) -> dict[str, float]:
    """Normalizes token usage dict to a standard format."""
    if not raw:
        return {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0}
    input_tokens = raw.get("input_tokens") or raw.get("prompt_tokens") or 0.0
    output_tokens = raw.get("output_tokens") or raw.get("completion_tokens") or 0.0
    total_tokens = raw.get("total_tokens") or (input_tokens + output_tokens)
    return {
        "input_tokens": float(input_tokens),
        "output_tokens": float(output_tokens),
        "total_tokens": float(total_tokens),
    }


def record_end(
    name: str,
    provider: str,
    model: str,
    start_time: float,
    usage: dict[str, float] | None,
) -> dict:
    """Records node execution metrics and returns state updates.

    Args:
        name: Node name.
        provider: Provider used (openai, gemini, ollama).
        model: Model name used.
        start_time: perf_counter value at node start.
        usage: Raw token usage from LLM response.

    Returns:
        Dict with trace, executed_agents, and token_usage for state update.
    """
    duration = round(time.perf_counter() - start_time, 4)

    norm = normalize_usage(usage)

    trace_entry = {
        "node": name,
        "provider": provider,
        "model": model,
        "duration": duration,
        "input_tokens": int(norm["input_tokens"]),
        "output_tokens": int(norm["output_tokens"]),
        "total_tokens": int(norm["total_tokens"]),
    }

    executed = [name] if name != "planner" else []

    return {
        "trace": [trace_entry],
        "executed_agents": executed,
        "token_usage": norm,
    }
