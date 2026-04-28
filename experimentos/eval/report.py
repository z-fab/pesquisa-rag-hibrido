"""Report module for rich terminal display of evaluation results."""

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table


def load_results(path: Path) -> dict:
    """Load evaluation results from a JSON file."""
    with open(path) as f:
        return json.load(f)


def build_summary_tables(data: dict) -> dict[str, Table]:
    """Build Rich Table objects from a results JSON dict.

    Returns a dict with keys: metadata, routing, confusion_matrix,
    retrieval, quality, detail, efficiency.
    """
    tables: dict[str, Table] = {}
    metadata = data.get("metadata", {})
    results = data.get("results", [])
    metrics = data.get("metrics", {})

    # --- metadata ---
    t = Table(title="Run Metadata")
    t.add_column("Param", style="bold")
    t.add_column("Value")
    for key, value in metadata.items():
        t.add_row(str(key), str(value))
    tables["metadata"] = t

    # --- routing ---
    routing = metrics.get("routing", {})
    t = Table(title="Routing")
    t.add_column("Metric", style="bold")
    t.add_column("Value")
    t.add_row("Accuracy", f"{routing.get('accuracy', 0):.2f}%")
    tables["routing"] = t

    # --- confusion_matrix ---
    cm = routing.get("confusion_matrix", {})
    types = ["S", "NS", "H"]
    t = Table(title="Confusion Matrix (Real \u2192 Predicted)")
    t.add_column("Real \\ Pred", style="bold")
    for tp in types:
        t.add_column(tp)
    for real in types:
        row = cm.get(real, {})
        t.add_row(real, *[str(row.get(pred, 0)) for pred in types])
    tables["confusion_matrix"] = t

    # --- retrieval ---
    retrieval = metrics.get("retrieval", {})
    t = Table(title="Retrieval Metrics")
    t.add_column("Type", style="bold")
    t.add_column("Exec. Accuracy")
    t.add_column("Answer Accuracy")
    t.add_column("Precision")
    t.add_column("Recall")
    for tp in types:
        r = retrieval.get(tp, {})
        t.add_row(
            tp,
            _fmt(r.get("execution_accuracy")),
            _fmt(r.get("answer_accuracy")),
            _fmt(r.get("precision")),
            _fmt(r.get("recall")),
        )
    tables["retrieval"] = t

    # --- quality ---
    quality = metrics.get("final_answer_quality", {})
    overall = quality.get("overall_avg", {})
    by_type = quality.get("by_type", {})
    t = Table(title="Final Answer Quality")
    t.add_column("Scope", style="bold")
    t.add_column("Completude")
    t.add_column("Fidelidade")
    t.add_column("Rastreabilidade")
    t.add_column("Media")
    t.add_row(
        "Overall",
        _fmt(overall.get("completude")),
        _fmt(overall.get("fidelidade")),
        _fmt(overall.get("rastreabilidade")),
        _fmt(overall.get("media")),
    )
    for tp in types:
        bt = by_type.get(tp, {})
        t.add_row(
            tp,
            _fmt(bt.get("completude")),
            _fmt(bt.get("fidelidade")),
            _fmt(bt.get("rastreabilidade")),
            _fmt(bt.get("media")),
        )
    tables["quality"] = t

    # --- detail ---
    t = Table(title="Per-Question Detail")
    t.add_column("ID", style="bold")
    t.add_column("Tipo")
    t.add_column("Rota OK?")
    t.add_column("SQL Match?")
    t.add_column("RAG Prec")
    t.add_column("RAG Rec")
    t.add_column("Compl.")
    t.add_column("Fidel.")
    t.add_column("Rastr.")
    t.add_column("Latencia")
    t.add_column("Tokens")
    for r in results:
        judgement = r.get("judgement", {})
        rag = r.get("output_rag", {})
        resp = judgement.get("response", {})
        sql_j = judgement.get("sql", {})
        latency = r.get("output_latency", {})
        tokens = r.get("output_token_usage", {})
        t.add_row(
            str(r.get("id", "")),
            str(r.get("type", "")),
            _bool_fmt(r.get("output_match_type")),
            _bool_fmt(sql_j.get("match")),
            _fmt(rag.get("precision")),
            _fmt(rag.get("recall")),
            _fmt(resp.get("completude")),
            _fmt(resp.get("fidelidade")),
            _fmt(resp.get("rastreabilidade")),
            _fmt(latency.get("total"), suffix="s"),
            _fmt(tokens.get("total_tokens")),
        )
    tables["detail"] = t

    # --- efficiency ---
    efficiency = metrics.get("efficiency", {})
    agent_latency = efficiency.get("avg_agent_latency", {})
    token_usage = efficiency.get("avg_token_usage", {})
    t = Table(title="Efficiency")
    t.add_column("Metric", style="bold")
    t.add_column("Value")
    t.add_row("Avg Total Latency", _fmt(efficiency.get("avg_total_latency"), suffix="s"))
    for agent, lat in agent_latency.items():
        t.add_row(f"  Avg {agent}", _fmt(lat, suffix="s"))
    for key, val in token_usage.items():
        t.add_row(f"Avg {key}", _fmt(val))
    tables["efficiency"] = t

    return tables


def display_report(data: dict) -> None:
    """Build and print all summary tables to the terminal."""
    console = Console()
    tables = build_summary_tables(data)
    for table in tables.values():
        console.print(table)
        console.print()


def display_comparative_report(datasets: list[dict]) -> None:
    """Display comparative tables across multiple evaluation runs.

    Each dataset should be a full results dict. One column per run_id is
    shown for: routing accuracy, quality dimensions, retrieval per type,
    and efficiency (latency + tokens).
    """
    console = Console()
    run_ids = [d.get("metadata", {}).get("run_id", f"run-{i}") for i, d in enumerate(datasets)]
    types = ["S", "NS", "H"]

    # --- Routing accuracy ---
    t = Table(title="Comparative: Routing Accuracy")
    t.add_column("Metric", style="bold")
    for rid in run_ids:
        t.add_column(rid)
    t.add_row(
        "Accuracy",
        *[f"{d.get('metrics', {}).get('routing', {}).get('accuracy', 0):.2f}%" for d in datasets],
    )
    console.print(t)
    console.print()

    # --- Quality dimensions ---
    t = Table(title="Comparative: Quality Dimensions")
    t.add_column("Dimension", style="bold")
    for rid in run_ids:
        t.add_column(rid)
    for dim in ["completude", "fidelidade", "rastreabilidade", "media"]:
        t.add_row(
            dim.capitalize(),
            *[
                _fmt(d.get("metrics", {}).get("final_answer_quality", {}).get("overall_avg", {}).get(dim))
                for d in datasets
            ],
        )
    console.print(t)
    console.print()

    # --- Retrieval per type ---
    t = Table(title="Comparative: Retrieval per Type")
    t.add_column("Type / Metric", style="bold")
    for rid in run_ids:
        t.add_column(rid)
    for tp in types:
        for metric in ["execution_accuracy", "answer_accuracy", "precision", "recall"]:
            vals = []
            for d in datasets:
                v = d.get("metrics", {}).get("retrieval", {}).get(tp, {}).get(metric)
                vals.append(_fmt(v))
            t.add_row(f"{tp} / {metric}", *vals)
    console.print(t)
    console.print()

    # --- Efficiency ---
    t = Table(title="Comparative: Efficiency")
    t.add_column("Metric", style="bold")
    for rid in run_ids:
        t.add_column(rid)
    t.add_row(
        "Avg Total Latency",
        *[_fmt(d.get("metrics", {}).get("efficiency", {}).get("avg_total_latency"), suffix="s") for d in datasets],
    )
    for key in ["input_tokens", "output_tokens", "total_tokens"]:
        t.add_row(
            f"Avg {key}",
            *[_fmt(d.get("metrics", {}).get("efficiency", {}).get("avg_token_usage", {}).get(key)) for d in datasets],
        )
    console.print(t)
    console.print()


def _fmt(value, suffix: str = "") -> str:
    """Format a value for display, handling None."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def _bool_fmt(value) -> str:
    """Format a boolean for display."""
    if value is None:
        return "-"
    return "\u2713" if value else "\u2717"
