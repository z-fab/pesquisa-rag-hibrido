"""Consolida resultados de todas as runs em CSVs agregados.

Produz dois artefatos em `<analysis_dir>/`:
- `aggregated.csv`      : uma linha por item (1.080+ linhas = N_runs × 30).
- `aggregated_runs.csv` : uma linha por run (métricas agregadas via eval/metrics.py).

Uso via CLI: `rag analyze`
Uso direto:  `uv run python -m scripts.aggregate`
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path
from statistics import mean, median

import numpy as np
from scipy import stats as scipy_stats

from eval.metrics import calculate_metrics

# Metadata dos modelos: directory key (nome da pasta em data/outputs/), display
# name, family (gemini/gpt/qwen), size (small/medium/large), is_edge (bool).

MODEL_META = {
    # Proprietários — Gemini
    "gemini-flash-lite": {"display": "Gemini Flash-Lite", "family": "gemini", "size": "small", "is_edge": False},
    "gemini-flash":      {"display": "Gemini Flash",      "family": "gemini", "size": "medium", "is_edge": False},
    "gemini-pro":        {"display": "Gemini Pro",        "family": "gemini", "size": "large",  "is_edge": False},
    # Proprietários — GPT
    "gpt-nano": {"display": "GPT-5 Nano", "family": "gpt", "size": "small", "is_edge": False},
    "gpt-mini": {"display": "GPT-5 Mini", "family": "gpt", "size": "medium", "is_edge": False},
    "gpt":      {"display": "GPT-5",      "family": "gpt", "size": "large",  "is_edge": False},
    # Open — Qwen 3.5 MoE (mesma geração)
    "qwen-35b":  {"display": "Qwen 3.5 35B-A3B",   "family": "qwen", "size": "small",  "is_edge": False},
    "qwen-122b": {"display": "Qwen 3.5 122B-A10B", "family": "qwen", "size": "medium", "is_edge": False},
    "qwen-397b": {"display": "Qwen 3.5 397B-A17B", "family": "qwen", "size": "large",  "is_edge": False},
    # Edge (analisados separadamente — modelos ≤10B dense, candidatos a deploy local)
    "ministral-3b": {"display": "Ministral 3B",    "family": "mistral", "size": "edge", "is_edge": True},
    "gemma3-4b":    {"display": "Gemma 3 4B",      "family": "gemma",   "size": "edge", "is_edge": True},
    "llama-8b":     {"display": "Llama 3.1 8B",    "family": "llama",   "size": "edge", "is_edge": True},
}

ARCH_DISPLAY = {
    "full": "Completa",
    "no-verifier": "Sem Verificação",
    "no-synthesizer": "Sem Síntese",
    "poc": "Simples",
}
ARCHES = list(ARCH_DISPLAY.keys())


# Bootstrap CI — exportado para uso em stats.py

def ci_95(values, stat_fn=np.mean, n_resamples: int = 10000, random_state: int = 42) -> tuple[float | None, float | None, float | None]:
    """Retorna (estimativa, ic_inferior, ic_superior) via bootstrap percentil.

    Não paramétrico — não assume normalidade. Apropriado para escalas 0-2
    discretas e proporções.
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if len(arr) == 0:
        return (None, None, None)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]), float(arr[0]))
    estimate = float(stat_fn(arr))
    try:
        result = scipy_stats.bootstrap(
            (arr,),
            statistic=stat_fn,
            n_resamples=n_resamples,
            confidence_level=0.95,
            method="percentile",
            random_state=random_state,
        )
        return (estimate, float(result.confidence_interval.low), float(result.confidence_interval.high))
    except Exception:
        return (estimate, None, None)


# Extração per-item (preserva todo o detalhe dos items)

def _arch_from_filename(name: str) -> str:
    for a in ARCHES:
        if f"_{a}_" in name:
            return a
    return "unknown"


_LEGACY_WARNED = False


def _judgements_from_evaluation(ev: dict) -> dict:
    """Normaliza o bloco de julgamento — padrão único é `judgements` + `judgement_agg`.

    Emite warning (uma vez) se encontrar formato legado (`judgement` sem `judgements`).
    Nesse caso, sugere rodar `rag rejudge` para migrar.
    """
    global _LEGACY_WARNED
    judgements = ev.get("judgements") or {}

    if not judgements:
        if ev.get("judgement") and not _LEGACY_WARNED:
            print("[warn] encontrado formato legado `judgement` sem `judgements`. "
                  "Rode `rag rejudge` para migrar. Este item será ignorado.")
            _LEGACY_WARNED = True
        return {"judgements": {}, "agg": {}, "n_judges": 0}

    agg = ev.get("judgement_agg") or {}
    return {"judgements": judgements, "agg": agg, "n_judges": len(judgements)}


def _dim_values(jd: dict, dim: str) -> tuple[float | None, list[int]]:
    """Retorna (média entre juízes, lista per-judge) para uma dimensão."""
    vals = []
    for j in jd["judgements"].values():
        v = j.get("response", {}).get(dim)
        if v is not None:
            vals.append(int(v))
    if not vals:
        return (None, [])
    return (mean(vals), vals)


def _sql_match_value(jd: dict) -> bool | None:
    """Retorna True/False/None via maioria entre juízes disponíveis."""
    votes = [j.get("sql", {}).get("match") for j in jd["judgements"].values() if j.get("sql")]
    votes = [v for v in votes if v is not None]
    if not votes:
        return None
    if len(votes) == 1:
        return bool(votes[0])
    return sum(votes) > len(votes) / 2


def extract_per_item_rows(model_key: str, results_path: Path) -> list[dict]:
    """Extrai uma linha por item do results_*.json."""
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    arch = _arch_from_filename(results_path.name)
    meta = MODEL_META[model_key]

    rows = []
    for r in data["results"]:
        jd = _judgements_from_evaluation(r)
        completude_mean, _ = _dim_values(jd, "completude")
        fidelidade_mean, _ = _dim_values(jd, "fidelidade")
        rastreabilidade_mean, _ = _dim_values(jd, "rastreabilidade")
        # Quality só é definido quando todas as 3 dimensões foram avaliadas.
        # Antes: tratava None como 0, enviesando quality para baixo em itens parcialmente avaliados.
        dims = [completude_mean, fidelidade_mean, rastreabilidade_mean]
        quality = sum(dims) / 3 if all(d is not None for d in dims) else None

        rag = r.get("output_rag") or {}
        latency = r.get("output_latency") or {}
        per_agent = latency.get("per_agent") or {}
        tokens = r.get("output_token_usage") or {}

        rows.append({
            "model_key": model_key,
            "model": meta["display"],
            "family": meta["family"],
            "size": meta["size"],
            "is_edge": meta["is_edge"],
            "architecture": arch,
            "architecture_display": ARCH_DISPLAY[arch],
            "item_id": r.get("id"),
            "type": r.get("type"),
            "n_judges": jd["n_judges"],
            # Síntese (3 dimensões)
            "completude": completude_mean,
            "fidelidade": fidelidade_mean,
            "rastreabilidade": rastreabilidade_mean,
            "quality": quality,
            # Recuperação (retrieval)
            "sql_match": _sql_match_value(jd),
            "retrieval_precision": rag.get("precision"),
            "retrieval_recall": rag.get("recall"),
            # Roteamento
            "type_predicted": r.get("output_type_predicted"),
            "match_type": r.get("output_match_type"),
            "match_agents": r.get("output_match_agents"),
            # Eficiência
            "latency_total": latency.get("total"),
            "latency_planner": per_agent.get("planner") or per_agent.get("simple_router"),
            "latency_sql": per_agent.get("sql_planner_executor"),
            "latency_rag": per_agent.get("text_retriever"),
            "latency_synthesizer": per_agent.get("synthesizer") or per_agent.get("simple_synthesizer"),
            "latency_verifier": per_agent.get("verifier"),
            "latency_consolidator": per_agent.get("consolidator") or per_agent.get("consolidator_lite"),
            "input_tokens": tokens.get("input_tokens"),
            "output_tokens": tokens.get("output_tokens"),
            "total_tokens": tokens.get("total_tokens"),
        })
    return rows


# Extração per-run (delegada ao eval/metrics.py)

def _run_summary_row(model_key: str, results_path: Path, per_item_rows: list[dict]) -> dict:
    """Resumo per-run usando calculate_metrics quando possível + CIs bootstrap."""
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    arch = _arch_from_filename(results_path.name)
    meta = MODEL_META[model_key]

    # Filtrar apenas linhas deste run (modelo específico × arquitetura)
    run_rows = [r for r in per_item_rows if r["architecture"] == arch and r["model_key"] == model_key]

    def values(field, filt=None):
        return [r[field] for r in run_rows if r[field] is not None and (filt is None or filt(r))]

    def mean_ci(field, filt=None):
        est, lo, hi = ci_95(values(field, filt))
        return est, lo, hi

    comp_est, comp_lo, comp_hi = mean_ci("completude")
    fid_est, fid_lo, fid_hi = mean_ci("fidelidade")
    rast_est, rast_lo, rast_hi = mean_ci("rastreabilidade")
    qual_est, qual_lo, qual_hi = mean_ci("quality")

    # Per-type quality
    qual_S = mean_ci("quality", lambda r: r["type"] == "S")
    qual_NS = mean_ci("quality", lambda r: r["type"] == "NS")
    qual_H = mean_ci("quality", lambda r: r["type"] == "H")

    # Métricas de roteamento
    type_match = [1 if r["match_type"] else 0 for r in run_rows if r["match_type"] is not None]
    agent_match = [1 if r["match_agents"] else 0 for r in run_rows if r["match_agents"] is not None]
    routing_acc, routing_lo, routing_hi = ci_95(type_match)
    agent_acc, agent_lo, agent_hi = ci_95(agent_match)

    # SQL match (apenas S e H)
    sql_mask = lambda r: r["type"] in ("S", "H") and r["sql_match"] is not None  # noqa: E731
    sql_vals = [1 if r["sql_match"] else 0 for r in run_rows if sql_mask(r)]
    sql_acc, sql_lo, sql_hi = ci_95(sql_vals)

    # Retrieval precision/recall (NS e H)
    retrieval_mask = lambda r: r["type"] in ("NS", "H") and r["retrieval_precision"] is not None  # noqa: E731
    prec_est, prec_lo, prec_hi = mean_ci("retrieval_precision", retrieval_mask)
    rec_est, rec_lo, rec_hi = mean_ci("retrieval_recall", retrieval_mask)

    # Eficiência
    lat_est, lat_lo, lat_hi = mean_ci("latency_total")
    lat_med = median(values("latency_total")) if values("latency_total") else None
    tokens_in = mean(values("input_tokens")) if values("input_tokens") else None
    tokens_out = mean(values("output_tokens")) if values("output_tokens") else None
    tokens_total = mean(values("total_tokens")) if values("total_tokens") else None

    return {
        "model_key": model_key,
        "model": meta["display"],
        "family": meta["family"],
        "size": meta["size"],
        "is_edge": meta["is_edge"],
        "architecture": arch,
        "architecture_display": ARCH_DISPLAY[arch],
        "n_items": len(run_rows),
        # Síntese — 3 dimensões + quality auxiliar
        "completude_mean": comp_est, "completude_ci_low": comp_lo, "completude_ci_high": comp_hi,
        "fidelidade_mean": fid_est, "fidelidade_ci_low": fid_lo, "fidelidade_ci_high": fid_hi,
        "rastreabilidade_mean": rast_est, "rastreabilidade_ci_low": rast_lo, "rastreabilidade_ci_high": rast_hi,
        "quality_mean": qual_est, "quality_ci_low": qual_lo, "quality_ci_high": qual_hi,
        "quality_S": qual_S[0], "quality_NS": qual_NS[0], "quality_H": qual_H[0],
        # Roteamento
        "routing_accuracy": routing_acc, "routing_ci_low": routing_lo, "routing_ci_high": routing_hi,
        "agent_match_rate": agent_acc, "agent_ci_low": agent_lo, "agent_ci_high": agent_hi,
        # SQL
        "sql_match_rate": sql_acc, "sql_ci_low": sql_lo, "sql_ci_high": sql_hi,
        # Retrieval
        "retrieval_precision_mean": prec_est, "retrieval_precision_ci_low": prec_lo, "retrieval_precision_ci_high": prec_hi,
        "retrieval_recall_mean": rec_est, "retrieval_recall_ci_low": rec_lo, "retrieval_recall_ci_high": rec_hi,
        # Eficiência
        "latency_mean": lat_est, "latency_ci_low": lat_lo, "latency_ci_high": lat_hi, "latency_median": lat_med,
        "input_tokens_mean": tokens_in,
        "output_tokens_mean": tokens_out,
        "total_tokens_mean": tokens_total,
    }


def run(outputs_dir: Path, analysis_dir: Path) -> tuple[Path, Path]:
    """Agrega todos os results_*.json em outputs_dir/<modelo>/ e grava CSVs em analysis_dir.

    Returns:
        (path_per_item_csv, path_per_run_csv)
    """
    analysis_dir.mkdir(parents=True, exist_ok=True)

    per_item_rows = []
    missing = []

    for model_key in MODEL_META:
        model_dir = outputs_dir / model_key
        if not model_dir.exists():
            missing.append(model_key)
            continue
        for results_path in sorted(model_dir.glob("results_*.json")):
            per_item_rows.extend(extract_per_item_rows(model_key, results_path))

    if missing:
        print(f"Aviso: pastas de modelos ausentes: {missing}")

    per_run_rows = []
    for model_key in MODEL_META:
        model_dir = outputs_dir / model_key
        if not model_dir.exists():
            continue
        for results_path in sorted(model_dir.glob("results_*.json")):
            per_run_rows.append(_run_summary_row(model_key, results_path, per_item_rows))

    path_items = analysis_dir / "aggregated.csv"
    path_runs = analysis_dir / "aggregated_runs.csv"

    if per_item_rows:
        with open(path_items, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_item_rows[0].keys()))
            w.writeheader()
            w.writerows(per_item_rows)
        print(f"Gravado: {path_items} ({len(per_item_rows)} linhas)")

    if per_run_rows:
        with open(path_runs, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
            w.writeheader()
            w.writerows(per_run_rows)
        print(f"Gravado: {path_runs} ({len(per_run_rows)} linhas)")

    return path_items, path_runs


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    run(root / "data" / "outputs", root / "data" / "outputs" / "_analysis")
