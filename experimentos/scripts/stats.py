"""Testes estatísticos sobre `aggregated.csv` e `aggregated_runs.csv`.

Estrutura conforme plano v2 (`docs/reports/plano_capitulo_resultados.md`):
- Parte 2: `full` vs `poc` pooled (Wilcoxon + McNemar).
- Parte 3: ablação sequencial (Verifier, Synthesizer, Router).
- Parte 4: Friedman por família e por tipo + Wilcoxon post-hoc.
- Parte 6: Cohen's κ weighted e ICC entre juízes (se houver J2/J3).

Produz em `<analysis_dir>/`:
- `stats.json`         : todos os resultados numéricos.
- `stats_summary.txt`  : resumo em português.

Uso via CLI: `rag analyze`
Uso direto: `uv run python -m scripts.stats`
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ARCHES = ["full", "no-verifier", "no-synthesizer", "poc"]


def _wilcoxon(a, b, label=""):
    """Wilcoxon pareado + rank-biserial effect size."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = (~np.isnan(a)) & (~np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2 or np.all(a == b):
        return {"label": label, "n": int(len(a)), "p_value": None, "statistic": None,
                "rank_biserial": 0.0, "mean_diff": float(np.mean(a - b)) if len(a) else None,
                "median_diff": float(np.median(a - b)) if len(a) else None, "note": "degenerate"}
    try:
        res = scipy_stats.wilcoxon(a, b, zero_method="pratt", alternative="two-sided")
        diff = a - b
        n = len(diff)
        # Rank-biserial: (positive ranks - negative ranks) / total rank sum
        abs_diff = np.abs(diff[diff != 0])
        ranks = scipy_stats.rankdata(abs_diff)
        signs = np.sign(diff[diff != 0])
        pos_ranks = ranks[signs > 0].sum()
        neg_ranks = ranks[signs < 0].sum()
        total_ranks = ranks.sum()
        r_rb = (pos_ranks - neg_ranks) / total_ranks if total_ranks > 0 else 0.0
        return {
            "label": label, "n": int(n),
            "p_value": float(res.pvalue),
            "statistic": float(res.statistic),
            "rank_biserial": float(r_rb),
            "mean_diff": float(np.mean(diff)),
            "median_diff": float(np.median(diff)),
        }
    except Exception as e:
        return {"label": label, "n": int(len(a)), "p_value": None, "note": f"error: {e}"}


def _friedman(*samples, label=""):
    """Friedman test para k amostras pareadas."""
    arrs = [np.asarray(s, dtype=float) for s in samples]
    arrs = [a[~np.isnan(a)] for a in arrs]
    # Truncar para o menor tamanho (pareado)
    min_n = min(len(a) for a in arrs) if arrs else 0
    if min_n < 3:
        return {"label": label, "n": min_n, "p_value": None, "note": "insufficient data"}
    arrs = [a[:min_n] for a in arrs]
    try:
        res = scipy_stats.friedmanchisquare(*arrs)
        # Kendall's W coefficient of concordance
        k = len(arrs)
        n = min_n
        chi2 = res.statistic
        w = chi2 / (n * (k - 1))
        return {
            "label": label, "n": int(n), "k": k,
            "statistic": float(chi2),
            "p_value": float(res.pvalue),
            "kendall_w": float(w),
        }
    except Exception as e:
        return {"label": label, "n": int(min_n), "p_value": None, "note": f"error: {e}"}


def _mcnemar(a, b, label=""):
    """McNemar exato (pareado binário). a, b são listas de 0/1."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    mask = (a >= 0) & (b >= 0)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 2:
        return {"label": label, "n": n, "p_value": None, "note": "insufficient data"}
    b_count = int(((a == 1) & (b == 0)).sum())  # a-only
    c_count = int(((a == 0) & (b == 1)).sum())  # b-only
    if b_count + c_count == 0:
        return {"label": label, "n": n, "p_value": 1.0, "b": b_count, "c": c_count, "note": "no discordances"}
    p = scipy_stats.binomtest(min(b_count, c_count), b_count + c_count, p=0.5, alternative="two-sided").pvalue
    return {
        "label": label, "n": int(n),
        "b_a_only": b_count, "c_b_only": c_count,
        "p_value": float(p),
        "prop_a": float(a.mean()), "prop_b": float(b.mean()),
    }


def _cohens_kappa_weighted(a, b, weights: str = "quadratic"):
    """Cohen's kappa weighted para escalas ordinais (usa sklearn).

    Pesos quadráticos são o padrão em escalas ordinais curtas — discordâncias de
    2 níveis contam 4× mais que discordâncias de 1 nível.
    """
    from sklearn.metrics import cohen_kappa_score

    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    mask = (a >= 0) & (b >= 0)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return None
    try:
        return float(cohen_kappa_score(a, b, weights=weights))
    except ValueError:
        return None


def _icc_3k(data):
    """ICC(3,k) via pingouin — two-way mixed, consistency, average measures.

    Corresponde a `ICC(C,k)` na notação do pingouin (Shrout-Fleiss 1979,
    forma consistency, k avaliadores fixos). Fórmula canônica:
    ICC(3,k) = (MSR − MSE) / MSR.

    data: array (n_subjects, n_raters).
    """
    import pandas as pd
    import pingouin as pg

    arr = np.asarray(data, dtype=float)
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    n, k = arr.shape
    long = pd.DataFrame({
        "item": np.repeat(np.arange(n), k),
        "rater": np.tile(np.arange(k), n),
        "score": arr.flatten(),
    })
    try:
        res = pg.intraclass_corr(data=long, targets="item", raters="rater", ratings="score")
        icc_row = res[res["Type"] == "ICC(C,k)"]["ICC"]
        return float(icc_row.iloc[0]) if len(icc_row) else None
    except Exception:
        return None


def _direction_consistency(df_items: pd.DataFrame, arch_a: str, arch_b: str, metric: str) -> dict:
    """Retorna fração de modelos em que arch_a > arch_b na métrica dada."""
    by_model = df_items.groupby(["model_key", "architecture"])[metric].mean().unstack("architecture")
    if arch_a not in by_model or arch_b not in by_model:
        return {"n_models": 0, "a_wins": 0, "b_wins": 0, "ties": 0}
    diff = by_model[arch_a] - by_model[arch_b]
    a_wins = int((diff > 0).sum())
    b_wins = int((diff < 0).sum())
    ties = int((diff == 0).sum())
    n = len(diff.dropna())
    return {
        "n_models": n,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "a_wins_pct": round(a_wins / n, 3) if n else 0,
    }


def run(analysis_dir: Path, outputs_dir: Path | None = None) -> Path:
    """Executa todos os testes sobre aggregated.csv + aggregated_runs.csv.

    Lê snapshots diretamente de `outputs_dir` para cálculos de concordância
    entre juízes (que exigem acesso aos julgamentos individuais, não só agregado).
    Se `outputs_dir` não for passado, deriva como `analysis_dir.parent`.

    Grava stats.json e stats_summary.txt em analysis_dir.
    """
    analysis_dir = Path(analysis_dir)
    outputs_dir = Path(outputs_dir) if outputs_dir else analysis_dir.parent
    df_items = pd.read_csv(analysis_dir / "aggregated.csv")
    df_runs = pd.read_csv(analysis_dir / "aggregated_runs.csv")

    # Filtrar apenas modelos não-Edge para análise principal
    df_main = df_items[~df_items["is_edge"]].copy()

    out = {
        "metadata": {
            "n_items_main": int(len(df_main)),
            "n_items_edge": int(df_items["is_edge"].sum()),
            "n_models_main": int(df_main["model_key"].nunique()),
            "n_models_edge": int(df_items[df_items["is_edge"]]["model_key"].nunique()),
            "architectures": ARCHES,
        },
    }

    # ---------------------------------------------------------------
    # Parte 2 — Impacto: full vs poc pooled (+ direction consistency)
    # ---------------------------------------------------------------
    out["parte2_impacto_full_vs_poc"] = _full_vs_poc_pooled(df_main)

    # ---------------------------------------------------------------
    # Parte 3 — Ablação sequencial
    # ---------------------------------------------------------------
    out["parte3_ablacao_sequencial"] = _ablacao_sequencial(df_main)

    # ---------------------------------------------------------------
    # Parte 4 — Heterogeneidade: por família e por tipo
    # ---------------------------------------------------------------
    out["parte4_por_familia"] = _por_familia(df_main)
    out["parte4_por_tipo"] = _por_tipo(df_main)

    # ---------------------------------------------------------------
    # Parte 6 — Concordância entre juízes
    # ---------------------------------------------------------------
    out["parte6_concordancia_juizes"] = _concordancia_juizes(df_items, outputs_dir)

    # Gravar artefatos
    path_json = analysis_dir / "stats.json"
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"Gravado: {path_json}")

    path_txt = analysis_dir / "stats_summary.txt"
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(_format_summary(out))
    print(f"Gravado: {path_txt}")

    return path_json


def _pivot_by_item(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Retorna DataFrame (modelo, item_id) × arquitetura para uma métrica."""
    pivot = df.pivot_table(
        index=["model_key", "item_id"],
        columns="architecture",
        values=metric,
        aggfunc="mean",
    )
    return pivot.reindex(columns=ARCHES).dropna(how="any")


def _full_vs_poc_pooled(df: pd.DataFrame) -> dict:
    """Testes pareados full vs poc em cada métrica, pooled entre modelos."""
    results = {}
    metrics_continuous = [
        "quality", "completude", "fidelidade", "rastreabilidade",
        "retrieval_precision", "retrieval_recall", "latency_total",
    ]
    metrics_binary = ["match_type", "match_agents", "sql_match"]

    for metric in metrics_continuous:
        pivot = _pivot_by_item(df, metric)
        if "full" not in pivot.columns or "poc" not in pivot.columns:
            continue
        r = _wilcoxon(pivot["full"].values, pivot["poc"].values, label=f"full vs poc — {metric}")
        r["direction"] = _direction_consistency(df, "full", "poc", metric)
        results[metric] = r

    for metric in metrics_binary:
        pivot = _pivot_by_item(df, metric).astype(int)
        if "full" not in pivot.columns or "poc" not in pivot.columns:
            continue
        r = _mcnemar(pivot["full"].values, pivot["poc"].values, label=f"full vs poc — {metric}")
        results[metric] = r

    # Friedman global entre as 4 arquiteturas (quality)
    pivot_q = _pivot_by_item(df, "quality")
    if all(a in pivot_q.columns for a in ARCHES):
        results["friedman_4arch_quality"] = _friedman(
            *[pivot_q[a].values for a in ARCHES], label="Friedman — 4 arquiteturas, quality"
        )

    return results


def _ablacao_sequencial(df: pd.DataFrame) -> dict:
    """Decomposição sequencial: Verifier (full vs no-ver), Synth (no-ver vs no-syn), Router (no-syn vs poc)."""
    results = {}

    def ablation(arch_a, arch_b, metrics, label):
        sub = {"label": label, "comparacao": f"{arch_a} vs {arch_b}", "metricas": {}}
        for metric in metrics:
            pivot = _pivot_by_item(df, metric)
            if arch_a not in pivot.columns or arch_b not in pivot.columns:
                continue
            sub["metricas"][metric] = _wilcoxon(
                pivot[arch_a].values, pivot[arch_b].values, label=f"{arch_a} vs {arch_b} — {metric}"
            )
        return sub

    # Verifier
    results["verifier"] = ablation(
        "full", "no-verifier",
        ["quality", "completude", "fidelidade", "rastreabilidade",
         "sql_match", "retrieval_precision", "retrieval_recall", "latency_total"],
        label="Contribuição do Verifier",
    )
    # Synthesizer — só métricas de síntese + latência (upstream não muda)
    results["synthesizer"] = ablation(
        "no-verifier", "no-synthesizer",
        ["quality", "completude", "fidelidade", "rastreabilidade", "latency_total"],
        label="Contribuição do Synthesizer",
    )
    # Router — métricas de roteamento + downstream
    results["router"] = ablation(
        "no-synthesizer", "poc",
        ["quality", "match_type", "match_agents", "sql_match",
         "retrieval_precision", "retrieval_recall", "latency_total"],
        label="Contribuição do Router",
    )
    return results


def _por_familia(df: pd.DataFrame) -> dict:
    """Friedman por família + Wilcoxon post-hoc full vs poc intra-família."""
    results = {}
    for family, sub in df.groupby("family"):
        pivot = _pivot_by_item(sub, "quality")
        fried = _friedman(*[pivot[a].values for a in ARCHES if a in pivot.columns],
                          label=f"Friedman — família {family}")
        post_hoc = None
        if fried.get("p_value") is not None and fried["p_value"] < 0.05 and "full" in pivot.columns and "poc" in pivot.columns:
            post_hoc = _wilcoxon(pivot["full"].values, pivot["poc"].values,
                                 label=f"Wilcoxon post-hoc — {family} full vs poc")
        results[family] = {"friedman": fried, "wilcoxon_full_vs_poc": post_hoc}
    return results


def _por_tipo(df: pd.DataFrame) -> dict:
    """Friedman por tipo de pergunta (S/NS/H) + Wilcoxon post-hoc full vs poc."""
    results = {}
    for tipo in ["S", "NS", "H"]:
        sub = df[df["type"] == tipo]
        pivot = _pivot_by_item(sub, "quality")
        fried = _friedman(*[pivot[a].values for a in ARCHES if a in pivot.columns],
                          label=f"Friedman — tipo {tipo}")
        post_hoc = None
        if fried.get("p_value") is not None and fried["p_value"] < 0.05 and "full" in pivot.columns and "poc" in pivot.columns:
            post_hoc = _wilcoxon(pivot["full"].values, pivot["poc"].values,
                                 label=f"Wilcoxon post-hoc — {tipo} full vs poc")
        results[tipo] = {"friedman": fried, "wilcoxon_full_vs_poc": post_hoc}
    return results


def _concordancia_juizes(df: pd.DataFrame, outputs_dir: Path) -> dict:
    """Cohen's κ weighted entre pares de juízes + ICC(3,k) para o score agregado.

    Lê os snapshots diretamente para acessar `judgements.j1/j2/j3` individuais,
    já que o CSV agregado só tem a média. Calcula:
    - κ weighted por par (j1-j2, j1-j3, j2-j3) × dimensão (completude, fidelidade, rastreabilidade)
    - ICC(3,k) sobre o score agregado (média das 3 dimensões por juiz)
    - Contagem de itens com divergência alta (|max - min| > 1 em alguma dimensão)
    """
    dims = ("completude", "fidelidade", "rastreabilidade")
    pairs = [("j1", "j2"), ("j1", "j3"), ("j2", "j3")]

    # Coleta triplas de avaliações por item (só itens com todos os 3 juízes presentes)
    # Restringe ao pool principal (9 modelos × 4 arquiteturas × 30 itens = 1.080 principais
    # + 3 edge × 4 × 30 = 360 edge = 1.440 itens). Exclui experimentos em _mix.
    items = []
    for snap_path in outputs_dir.rglob("snapshot_*.json"):
        if any(p in snap_path.parts for p in ("_analysis", "_mix")):
            continue
        # Pula arquivos auxiliares de experimento (ex.: .verifier_signal.json que contém dict, não lista)
        if snap_path.name.endswith(".verifier_signal.json"):
            continue
        try:
            with open(snap_path, encoding="utf-8") as f:
                snapshots = json.load(f)
        except Exception:
            continue
        # Proteção adicional: se o conteúdo não for lista de itens, pula
        if not isinstance(snapshots, list):
            continue
        for s in snapshots:
            judgements = s.get("evaluation", {}).get("judgements", {})
            if not all(j in judgements for j in ("j1", "j2", "j3")):
                continue
            entry = {}
            complete = True
            for j in ("j1", "j2", "j3"):
                resp = judgements[j].get("response") or {}
                scores = {d: resp.get(d) for d in dims}
                if any(v is None for v in scores.values()):
                    complete = False
                    break
                entry[j] = scores
            if complete:
                items.append(entry)

    n = len(items)
    if n < 2:
        return {
            "status": "sem dados suficientes (nenhum item com 3 juízes completos)",
            "n_items": n,
        }

    # Cohen's κ weighted por par × dimensão
    kappas = {}
    for j_a, j_b in pairs:
        kappas[f"{j_a}_vs_{j_b}"] = {}
        for d in dims:
            a = [it[j_a][d] for it in items]
            b = [it[j_b][d] for it in items]
            k = _cohens_kappa_weighted(a, b)
            kappas[f"{j_a}_vs_{j_b}"][d] = {
                "kappa_weighted": k,
                "interpretation": _kappa_label(k),
            }

    # ICC(3,k) sobre quality agregado (média das 3 dimensões por juiz)
    matrix = []
    for it in items:
        row = [(it[j]["completude"] + it[j]["fidelidade"] + it[j]["rastreabilidade"]) / 3 for j in ("j1", "j2", "j3")]
        matrix.append(row)
    icc_value = _icc_3k(matrix)

    # Itens com alta divergência
    high_div = 0
    for it in items:
        for d in dims:
            vals = [it[j][d] for j in ("j1", "j2", "j3")]
            if max(vals) - min(vals) >= 2:
                high_div += 1
                break

    return {
        "status": "calculado",
        "n_items": n,
        "cohens_kappa_weighted": kappas,
        "icc_3k_quality": {
            "value": icc_value,
            "interpretation": _icc_label(icc_value),
        },
        "high_divergence_items": {
            "count": high_div,
            "pct": round(100 * high_div / n, 1) if n else 0,
            "criterio": "|max - min| >= 2 em alguma dimensão",
        },
    }


def _kappa_label(k: float | None) -> str:
    if k is None:
        return "n/a"
    if k >= 0.80:
        return "quase perfeita"
    if k >= 0.60:
        return "substancial"
    if k >= 0.40:
        return "moderada"
    if k >= 0.20:
        return "fraca"
    return "muito fraca/ausente"


def _icc_label(i: float | None) -> str:
    if i is None:
        return "n/a"
    if i >= 0.90:
        return "excelente"
    if i >= 0.75:
        return "boa"
    if i >= 0.50:
        return "moderada"
    return "fraca"


def _format_summary(out: dict) -> str:
    """Resumo legível em português."""
    lines = ["Resumo estatístico — Plano v2", "=" * 50, ""]
    meta = out["metadata"]
    lines.append(f"n_items main = {meta['n_items_main']}, n_models_main = {meta['n_models_main']}")
    lines.append(f"n_items edge = {meta['n_items_edge']}, n_models_edge = {meta['n_models_edge']}")
    lines.append("")

    # Parte 2
    lines.append("[Parte 2] Impacto — Completa vs Simples (pooled)")
    for metric, r in out["parte2_impacto_full_vs_poc"].items():
        if metric.startswith("friedman"):
            lines.append(f"   {metric}: chi2={r.get('statistic'):.2f}, p={r.get('p_value'):.4g}, W={r.get('kendall_w'):.3f}")
        elif "p_value" in r and r.get("p_value") is not None:
            p = r["p_value"]
            star = "[SIG]" if p < 0.05 else "[ns]"
            mean_diff = r.get("mean_diff") or 0
            r_rb = r.get("rank_biserial") or 0
            dir_info = r.get("direction", {})
            extra = f" direção={dir_info.get('a_wins', 0)}/{dir_info.get('n_models', 0)} modelos" if dir_info else ""
            lines.append(f"   {metric:<28s}: p={p:.4g} mean_diff={mean_diff:+.3f} r_rb={r_rb:+.2f}{extra} {star}")
    lines.append("")

    # Parte 3
    lines.append("[Parte 3] Ablação sequencial")
    for componente in ("verifier", "synthesizer", "router"):
        blk = out["parte3_ablacao_sequencial"].get(componente, {})
        lines.append(f"   {componente} ({blk.get('comparacao', '')}):")
        for metric, r in blk.get("metricas", {}).items():
            if r.get("p_value") is None:
                continue
            p = r["p_value"]
            star = "[SIG]" if p < 0.05 else "[ns]"
            lines.append(f"      {metric:<26s}: p={p:.4g} mean_diff={r.get('mean_diff') or 0:+.3f} r_rb={r.get('rank_biserial') or 0:+.2f} {star}")
    lines.append("")

    # Parte 4
    lines.append("[Parte 4] Heterogeneidade — por família")
    for fam, blk in out["parte4_por_familia"].items():
        fried = blk["friedman"]
        if fried.get("p_value") is not None:
            star = "[SIG]" if fried["p_value"] < 0.05 else "[ns]"
            lines.append(f"   {fam:<10s} Friedman: chi2={fried.get('statistic', 0):.2f} p={fried['p_value']:.4g} W={fried.get('kendall_w', 0):.3f} {star}")

    lines.append("")
    lines.append("[Parte 4] Heterogeneidade — por tipo de pergunta")
    for tipo, blk in out["parte4_por_tipo"].items():
        fried = blk["friedman"]
        if fried.get("p_value") is not None:
            star = "[SIG]" if fried["p_value"] < 0.05 else "[ns]"
            lines.append(f"   {tipo:<4s} Friedman: chi2={fried.get('statistic', 0):.2f} p={fried['p_value']:.4g} W={fried.get('kendall_w', 0):.3f} {star}")

    lines.append("")
    juiz = out["parte6_concordancia_juizes"]
    lines.append(f"[Parte 6] Concordância entre juízes: {juiz.get('status', '?')}")
    if juiz.get("n_items"):
        lines.append(f"   n itens com 3 juízes completos: {juiz['n_items']}")
    if juiz.get("cohens_kappa_weighted"):
        lines.append("   Cohen's κ weighted (pares × dimensões):")
        for pair, dims in juiz["cohens_kappa_weighted"].items():
            for dim, info in dims.items():
                k = info["kappa_weighted"]
                k_str = f"{k:+.3f}" if k is not None else "n/a"
                lines.append(f"      {pair:10s}  {dim:20s}  κ={k_str}  ({info['interpretation']})")
    if juiz.get("icc_3k_quality"):
        icc = juiz["icc_3k_quality"]
        icc_str = f"{icc['value']:.3f}" if icc["value"] is not None else "n/a"
        lines.append(f"   ICC(3,k) quality: {icc_str}  ({icc['interpretation']})")
    if juiz.get("high_divergence_items"):
        hd = juiz["high_divergence_items"]
        lines.append(f"   Itens com alta divergência ({hd['criterio']}): {hd['count']} ({hd['pct']}%)")
    if juiz.get("nota"):
        lines.append(f"   {juiz['nota']}")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    run(root / "data" / "outputs" / "_analysis")
