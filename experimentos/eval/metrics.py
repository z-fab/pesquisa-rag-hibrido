from collections import defaultdict

from rich.console import Console
from rich.table import Table

console = Console()


def calculate_metrics(evaluation_results: list) -> dict:
    """Calculates aggregated metrics from evaluation results."""
    labels = ["S", "NS", "H"]
    confusion = {true: {pred: 0 for pred in labels} for true in labels}

    metrics = {
        "total_questions": len(evaluation_results),
        "routing": {"accuracy": 0.0, "confusion_matrix": confusion},
        "retrieval": {
            "S": {"execution_accuracy": 0.0, "answer_accuracy": 0.0},
            "NS": {"precision": 0.0, "recall": 0.0},
            "H": {
                "execution_accuracy": 0.0,
                "answer_accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            },
        },
        "final_answer_quality": {
            "overall_avg": {
                "completude": 0.0,
                "fidelidade": 0.0,
                "rastreabilidade": 0.0,
                "media": 0.0,
            },
            "by_type": {
                t: {
                    "completude": 0.0,
                    "fidelidade": 0.0,
                    "rastreabilidade": 0.0,
                    "media": 0.0,
                    "count": 0,
                }
                for t in labels
            },
        },
        "efficiency": {
            "avg_total_latency": 0.0,
            "avg_agent_latency": {},
            "avg_token_usage": {
                "input_tokens": 0.0,
                "output_tokens": 0.0,
                "total_tokens": 0.0,
            },
        },
    }

    acc = defaultdict(int)
    acc["agent_total_latency"] = defaultdict(float)
    acc["agent_count"] = defaultdict(int)
    acc["quality"] = {"completude": 0.0, "fidelidade": 0.0, "rastreabilidade": 0.0, "media": 0.0}

    n = len(evaluation_results)
    if n == 0:
        return metrics

    for res in evaluation_results:
        t = res["type"]
        pred = res["output_type_predicted"]
        if t in labels and pred in labels:
            metrics["routing"]["confusion_matrix"][t][pred] += 1
        acc["routing_hits"] += int(res["output_match_type"])

        acc["total_latency"] += res.get("output_latency", {}).get("total", 0.0)
        for agent, lat in res.get("output_latency", {}).get("per_agent", {}).items():
            acc["agent_total_latency"][agent] += lat
            acc["agent_count"][agent] += 1

        tu = res.get("output_token_usage", {})
        acc["input_tokens"] += tu.get("input_tokens", 0)
        acc["output_tokens"] += tu.get("output_tokens", 0)
        acc["total_tokens"] += tu.get("total_tokens", 0)

        if t == "S":
            acc["S_total"] += 1
            sql_results = res.get("output_sql_results", [])
            executed = any(r.get("executed") for r in sql_results)
            if executed:
                acc["S_executed"] += 1
                if res.get("judgement", {}).get("sql", {}).get("match"):
                    acc["S_match"] += 1

        if t == "NS":
            acc["NS_total"] += 1
            prec = res.get("output_rag", {}).get("precision")
            rec = res.get("output_rag", {}).get("recall")
            if prec is not None:
                acc["NS_precision_sum"] += prec
                acc["NS_precision_count"] += 1
            if rec is not None:
                acc["NS_recall_sum"] += rec
                acc["NS_recall_count"] += 1

        if t == "H":
            acc["H_total"] += 1
            sql_results = res.get("output_sql_results", [])
            executed = any(r.get("executed") for r in sql_results)
            if executed:
                acc["H_sql_executed"] += 1
                if res.get("judgement", {}).get("sql", {}).get("match"):
                    acc["H_sql_match"] += 1
            prec = res.get("output_rag", {}).get("precision")
            rec = res.get("output_rag", {}).get("recall")
            if prec is not None:
                acc["H_rag_precision_sum"] += prec
                acc["H_rag_precision_count"] += 1
            if rec is not None:
                acc["H_rag_recall_sum"] += rec
                acc["H_rag_recall_count"] += 1

        rj = res.get("judgement", {}).get("response", {})
        for dim in ("completude", "fidelidade", "rastreabilidade"):
            acc["quality"][dim] += rj.get(dim, 0)
        acc["quality"]["media"] += rj.get("avg_score", 0)
        acc["quality_count"] += 1

        metrics["final_answer_quality"]["by_type"][t]["completude"] += rj.get("completude", 0)
        metrics["final_answer_quality"]["by_type"][t]["fidelidade"] += rj.get("fidelidade", 0)
        metrics["final_answer_quality"]["by_type"][t]["rastreabilidade"] += rj.get("rastreabilidade", 0)
        metrics["final_answer_quality"]["by_type"][t]["media"] += rj.get("avg_score", 0)
        metrics["final_answer_quality"]["by_type"][t]["count"] += 1

    metrics["routing"]["accuracy"] = round((acc["routing_hits"] / n) * 100, 2)

    metrics["efficiency"]["avg_total_latency"] = round(acc["total_latency"] / n, 2)
    metrics["efficiency"]["avg_agent_latency"] = {
        a: round(acc["agent_total_latency"][a] / acc["agent_count"][a], 2) for a in acc["agent_total_latency"]
    }
    metrics["efficiency"]["avg_token_usage"] = {
        "input_tokens": int(round(acc["input_tokens"] / n)),
        "output_tokens": int(round(acc["output_tokens"] / n)),
        "total_tokens": int(round(acc["total_tokens"] / n)),
    }

    if acc["S_total"]:
        metrics["retrieval"]["S"]["execution_accuracy"] = round((acc["S_executed"] / acc["S_total"]) * 100, 2)
        if acc["S_executed"]:
            metrics["retrieval"]["S"]["answer_accuracy"] = round((acc["S_match"] / acc["S_executed"]) * 100, 2)

    if acc["NS_total"]:
        if acc["NS_precision_count"]:
            metrics["retrieval"]["NS"]["precision"] = round(acc["NS_precision_sum"] / acc["NS_precision_count"], 2)
        if acc["NS_recall_count"]:
            metrics["retrieval"]["NS"]["recall"] = round(acc["NS_recall_sum"] / acc["NS_recall_count"], 2)

    if acc["H_total"]:
        metrics["retrieval"]["H"]["execution_accuracy"] = round((acc["H_sql_executed"] / acc["H_total"]) * 100, 2)
        if acc["H_sql_executed"]:
            metrics["retrieval"]["H"]["answer_accuracy"] = round((acc["H_sql_match"] / acc["H_sql_executed"]) * 100, 2)
        if acc["H_rag_precision_count"]:
            metrics["retrieval"]["H"]["precision"] = round(acc["H_rag_precision_sum"] / acc["H_rag_precision_count"], 2)
        if acc["H_rag_recall_count"]:
            metrics["retrieval"]["H"]["recall"] = round(acc["H_rag_recall_sum"] / acc["H_rag_recall_count"], 2)

    if acc["quality_count"]:
        qc = acc["quality_count"]
        metrics["final_answer_quality"]["overall_avg"] = {k: round(v / qc, 2) for k, v in acc["quality"].items()}

    for t_bucket in metrics["final_answer_quality"]["by_type"].values():
        c = t_bucket["count"]
        if c > 0:
            for dim in ("completude", "fidelidade", "rastreabilidade", "media"):
                t_bucket[dim] = round(t_bucket[dim] / c, 2)

    return metrics


def display_metrics(metrics: dict, metadata: dict) -> None:
    """Displays metrics using Rich tables."""
    console.print()
    console.rule("[bold]Evaluation Results[/bold]")

    meta_table = Table(title="Configuration")
    meta_table.add_column("Key", style="cyan")
    meta_table.add_column("Value", style="green")
    for k, v in metadata.items():
        meta_table.add_row(k, str(v))
    console.print(meta_table)

    console.print(f"\n[bold]Routing Accuracy:[/bold] {metrics['routing']['accuracy']}%")

    quality = metrics["final_answer_quality"]["overall_avg"]
    qt = Table(title="Answer Quality (Overall)")
    qt.add_column("Metric")
    qt.add_column("Score", justify="right")
    for k, v in quality.items():
        qt.add_row(k, str(v))
    console.print(qt)

    eff = metrics["efficiency"]
    console.print(f"\n[bold]Avg Latency:[/bold] {eff['avg_total_latency']}s")
    console.print(f"[bold]Avg Tokens:[/bold] {eff['avg_token_usage']['total_tokens']}")
