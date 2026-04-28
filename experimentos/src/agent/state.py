import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class SearchTask(BaseModel):
    """A single retrieval task produced by the Planner."""

    type: Literal["sql", "text"]
    query: str = Field(description="Reformulated sub-query for this source type")
    sources: list[str] = Field(default_factory=list, description="Suggested tables or document IDs")


class PlannerOutput(BaseModel):
    """Output of the Planner node: a list of search tasks."""

    searches: list[SearchTask]


class Reference(BaseModel):
    """A source reference attached to a synthesized segment."""

    source: str
    type: Literal["sql", "text"]
    section: str = ""


class SynthesizerSegment(BaseModel):
    """A segment of the synthesized answer with optional references."""

    text: str
    refs: list[Reference] = Field(default_factory=list)


class SynthesizerOutput(BaseModel):
    """Structured output schema for the synthesizer LLM call."""

    segments: list[SynthesizerSegment]


class SegmentVerdict(BaseModel):
    """Verification verdict for a single synthesized segment."""

    index: int
    verdict: Literal["supported", "not_supported", "partial"]
    reasoning: str


class CompletenessCheck(BaseModel):
    """Assessment of whether the answer covers the original question."""

    covered: bool
    missing_aspects: list[str] = Field(default_factory=list)


class VerifierOutput(BaseModel):
    """Structured output schema for the verifier LLM call."""

    segments: list[SegmentVerdict]
    completeness: CompletenessCheck
    overall_pass: bool
    feedback: str


def add_token_usage(left: dict[str, float] | None, right: dict[str, float] | None) -> dict[str, float]:
    """Custom reducer to sum token usage dictionaries."""
    if not left:
        left = {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0}
    if not right:
        right = {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0}
    return {
        "input_tokens": left.get("input_tokens", 0.0) + right.get("input_tokens", 0.0),
        "output_tokens": left.get("output_tokens", 0.0) + right.get("output_tokens", 0.0),
        "total_tokens": left.get("total_tokens", 0.0) + right.get("total_tokens", 0.0),
    }


class AgentState(TypedDict, total=False):
    # Input
    question: str

    # Planner outputs
    needs_retrieval: bool
    planner_output: PlannerOutput

    # SQL Planner & Executor outputs (list — one entry per sql SearchTask)
    sql_results: Annotated[list[dict], operator.add]

    # Text Retriever outputs (list — one entry per text SearchTask)
    text_results: Annotated[list[dict], operator.add]

    # Synthesizer outputs
    synthesizer_output: SynthesizerOutput

    # Verifier outputs
    verifier_output: VerifierOutput

    # Consolidator outputs
    final_answer: str

    # Flow control
    retry_count: int

    # Tracking
    trace: Annotated[list[dict], operator.add]
    executed_agents: Annotated[list[str], operator.add]
    token_usage: Annotated[dict[str, float], add_token_usage]
    total_latency: float
    total_start: float
