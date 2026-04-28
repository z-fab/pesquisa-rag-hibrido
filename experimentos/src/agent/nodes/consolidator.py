import time

from loguru import logger

from src.agent.state import AgentState
from src.utils.tracking import record_end


def consolidator_node(state: AgentState) -> dict:
    """Consolidator node: formats synthesized segments into text with numbered references."""
    logger.info("Executing Consolidator...")
    start_time = time.perf_counter()

    synth_output = state.get("synthesizer_output")
    segments = synth_output.segments if synth_output else []

    if not segments:
        logger.debug("No segments to consolidate")
        tracking = record_end("consolidator", "", "", start_time, None)
        return {"final_answer": "", **tracking}

    # 1. Collect unique refs and assign numbers
    ref_map: dict[tuple[str, str, str], int] = {}
    for seg in segments:
        for ref in seg.refs:
            key = (ref.source, ref.type, ref.section)
            if key not in ref_map:
                ref_map[key] = len(ref_map) + 1

    # 2. Build text with [n] markers
    parts = []
    for seg in segments:
        text = seg.text
        if seg.refs:
            markers = "".join(f"[{ref_map[(r.source, r.type, r.section)]}]" for r in seg.refs)
            text = f"{text} {markers}"
        parts.append(text)

    body = "\n\n".join(parts)

    # 3. Build reference list
    if ref_map:
        ref_lines = []
        for (source, type_, section), num in sorted(ref_map.items(), key=lambda x: x[1]):
            if type_ == "sql":
                ref_lines.append(f"[{num}] {source} (SQL)")
            else:
                line = f"[{num}] {source}"
                if section:
                    line += f" \u2014 {section}"
                ref_lines.append(line)
        final_answer = f"{body}\n\nReferências:\n" + "\n".join(ref_lines)
    else:
        final_answer = body

    logger.debug(f"Consolidated {len(segments)} segments with {len(ref_map)} unique refs")

    tracking = record_end("consolidator", "", "", start_time, None)
    return {
        "final_answer": final_answer,
        **tracking,
    }
