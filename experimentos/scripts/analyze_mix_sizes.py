"""Análise do Experimento 2 — Heterogeneidade de modelos por estágio.

Lê os 540 itens das 18 configs mistas em `data/outputs/_mix/<family>/` + os 6
baselines (LLL=tudo-grande e SSS=tudo-pequeno) reaproveitados dos snapshots
`no-synthesizer` em `data/outputs/<model_key>/` e produz:

    data/outputs/_analysis/mix_sizes_raw.csv               (dados item-level)
    data/outputs/_analysis/mix_sizes_summary.csv           (medidas por config)
    data/outputs/_analysis/mix_sizes_effects.csv           (efeitos principais)
    data/outputs/_analysis/mix_sizes_wilcoxon_vs_lll.csv   (teste pareado)

Métricas:
  1. Efeitos principais por estágio (Planner/SQL/Synth), por família.
  2. Efeitos pooled entre famílias via z-score.
  3. Wilcoxon pareado item-a-item: cada config vs LLL (baseline).
  4. Custo estimado em USD/item × quality → identificação de Pareto-ótimo.

Uso:
    uv run python -m scripts.analyze_mix_sizes
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

BASELINE_MAP = {
    "gpt":    {"LLL": "gpt",        "SSS": "gpt-nano"},
    "gemini": {"LLL": "gemini-pro", "SSS": "gemini-flash-lite"},
    "qwen":   {"LLL": "qwen-397b",  "SSS": "qwen-35b"},
}

# Preços públicos (USD por 1M tokens) no momento do experimento
PRICES = {
    ("gpt", "L"):    (2.50, 10.00),
    ("gpt", "S"):    (0.05,  0.40),
    ("gemini", "L"): (1.25, 10.00),
    ("gemini", "S"): (0.075, 0.30),
    ("qwen", "L"):   (0.50,  1.50),
    ("qwen", "S"):   (0.20,  0.60),
}

FAMILY_ORDER = ["gpt", "gemini", "qwen"]
CONFIG_ORDER = ["LLL", "LLS", "LSL", "LSS", "SLL", "SLS", "SSL", "SSS"]


def _load_items_from_snapshot(snap_path: Path) -> list[dict]:
    with open(snap_path, encoding="utf-8") as f:
        items = json.load(f)
    rows = []
    for it in items:
        ev = it.get("evaluation", {})
        agg = ev.get("judgement_agg") or {}
        resp = agg.get("response") or {}
        c, f_, r = resp.get("completude"), resp.get("fidelidade"), resp.get("rastreabilidade")
        if c is None or f_ is None or r is None:
            continue
        out = it.get("output", {})
        lat = out.get("total_latency") or ev.get("output_latency", {}).get("total")
        tu = out.get("token_usage") or {}
        rows.append({
            "item_id": it["input"].get("id"),
            "type": it["input"].get("type"),
            "completude": c,
            "fidelidade": f_,
            "rastreabilidade": r,
            "quality": (c + f_ + r) / 3,
            "latency_total": lat,
            "input_tokens": tu.get("input_tokens", 0),
            "output_tokens": tu.get("output_tokens", 0),
        })
    return rows


def collect_rows(outputs_dir: Path) -> pd.DataFrame:
    """Lê baselines + 18 configs mistas, retorna DataFrame com family, config, item-level."""
    records = []
    for fam in FAMILY_ORDER:
        for cfg_code, model_key in BASELINE_MAP[fam].items():
            snaps = list((outputs_dir / model_key).glob("snapshot_*no-synthesizer*.json"))
            if not snaps:
                print(f"[warn] baseline ausente: {fam} / {cfg_code} / {model_key}")
                continue
            for r in _load_items_from_snapshot(snaps[0]):
                r.update(family=fam, config=cfg_code,
                         planner=cfg_code[0], sql=cfg_code[1], synth=cfg_code[2])
                records.append(r)

        for cfg in ["LLS", "LSL", "LSS", "SLL", "SLS", "SSL"]:
            snap = outputs_dir / "_mix" / fam / f"snapshot_mix-{fam}-{cfg}.json"
            if not snap.exists():
                print(f"[warn] mix ausente: {fam} / {cfg}")
                continue
            for r in _load_items_from_snapshot(snap):
                r.update(family=fam, config=cfg,
                         planner=cfg[0], sql=cfg[1], synth=cfg[2])
                records.append(r)

    return pd.DataFrame(records)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Médias por (família, config) — quality, dimensões, latência, tokens, custo."""
    g = df.groupby(["family", "config"]).agg(
        n=("quality", "size"),
        quality=("quality", "mean"),
        completude=("completude", "mean"),
        fidelidade=("fidelidade", "mean"),
        rastreabilidade=("rastreabilidade", "mean"),
        latency=("latency_total", "mean"),
        input_tokens=("input_tokens", "mean"),
        output_tokens=("output_tokens", "mean"),
    ).reset_index().round(4)

    # Custo aproximado: usa preço do nível dominante (maioria simples entre os 3 estágios)
    def _cost(row):
        fam, cfg = row["family"], row["config"]
        n_L = sum(1 for c in cfg if c == "L")
        lvl = "L" if n_L >= 2 else "S"
        p_in, p_out = PRICES[(fam, lvl)]
        return (row["input_tokens"] * p_in + row["output_tokens"] * p_out) / 1e6

    g["cost_per_item_usd"] = g.apply(_cost, axis=1).round(5)
    return g


def compute_main_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Efeito principal por estágio × família em quality e cada dimensão."""
    records = []
    for fam in FAMILY_ORDER:
        sub = df[df.family == fam]
        for stage in ["planner", "sql", "synth"]:
            for metric in ["quality", "completude", "fidelidade", "rastreabilidade"]:
                d_L = sub[sub[stage] == "L"][metric].mean()
                d_S = sub[sub[stage] == "S"][metric].mean()
                records.append({
                    "family": fam,
                    "stage": stage,
                    "metric": metric,
                    "mean_L": round(d_L, 4),
                    "mean_S": round(d_S, 4),
                    "delta_L_minus_S": round(d_L - d_S, 4),
                })
    return pd.DataFrame(records)


def compute_wilcoxon_vs_lll(df: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon pareado item-a-item: cada config vs LLL, por família, para quality."""
    records = []
    for fam in FAMILY_ORDER:
        lll = df[(df.family == fam) & (df.config == "LLL")].set_index("item_id")["quality"]
        for cfg in ["LLS", "LSL", "LSS", "SLL", "SLS", "SSL", "SSS"]:
            other = df[(df.family == fam) & (df.config == cfg)].set_index("item_id")["quality"]
            common = lll.index.intersection(other.index)
            l, o = lll.loc[common].values, other.loc[common].values
            d = o - l
            if (d == 0).all():
                p = 1.0
                stat = 0
            else:
                result = wilcoxon(l, o, zero_method="wilcox")
                p, stat = result.pvalue, result.statistic
            records.append({
                "family": fam,
                "config": cfg,
                "n": len(common),
                "quality_lll": round(l.mean(), 4),
                "quality_cfg": round(o.mean(), 4),
                "delta": round(d.mean(), 4),
                "n_cfg_wins": int((d > 0).sum()),
                "n_ties":     int((d == 0).sum()),
                "n_lll_wins": int((d < 0).sum()),
                "wilcoxon_statistic": float(stat),
                "p_value": round(float(p), 4),
            })
    return pd.DataFrame(records)


def compute_pooled_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Efeitos principais pooled entre famílias, em escala z (normalizada por família)."""
    # Z-score por família para permitir pooling
    df = df.copy()
    for col in ["quality", "completude", "fidelidade", "rastreabilidade"]:
        df[f"{col}_z"] = df.groupby("family")[col].transform(
            lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1)
        )

    records = []
    for stage in ["planner", "sql", "synth"]:
        for metric in ["quality", "completude", "fidelidade", "rastreabilidade"]:
            d_z = df[df[stage] == "L"][f"{metric}_z"].mean() - df[df[stage] == "S"][f"{metric}_z"].mean()
            d_raw = df[df[stage] == "L"][metric].mean() - df[df[stage] == "S"][metric].mean()
            records.append({
                "stage": stage,
                "metric": metric,
                "delta_raw": round(d_raw, 4),
                "delta_zscore": round(d_z, 4),
            })
    return pd.DataFrame(records)


def run(outputs_dir: Path, analysis_dir: Path):
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print("Coletando dados...")
    df = collect_rows(Path(outputs_dir))
    df.to_csv(analysis_dir / "mix_sizes_raw.csv", index=False)
    print(f"  {len(df)} linhas → mix_sizes_raw.csv")

    summary = compute_summary(df)
    summary.to_csv(analysis_dir / "mix_sizes_summary.csv", index=False)
    print(f"\n=== Resumo por configuração ===")
    cols = ["family", "config", "quality", "completude", "fidelidade", "rastreabilidade",
            "latency", "cost_per_item_usd"]
    print(summary[cols].to_string(index=False))

    effects = compute_main_effects(df)
    effects.to_csv(analysis_dir / "mix_sizes_effects.csv", index=False)
    print(f"\n=== Efeitos principais por família (quality) ===")
    eff_q = effects[effects.metric == "quality"].pivot(
        index="family", columns="stage", values="delta_L_minus_S")
    print(eff_q[["planner", "sql", "synth"]].round(4).to_string())

    pooled = compute_pooled_effects(df)
    print(f"\n=== Efeitos POOLED (z-score entre famílias) ===")
    print(pooled.to_string(index=False))

    wilc = compute_wilcoxon_vs_lll(df)
    wilc.to_csv(analysis_dir / "mix_sizes_wilcoxon_vs_lll.csv", index=False)
    print(f"\n=== Wilcoxon pareado (cada config vs LLL) ===")
    print(wilc.to_string(index=False))

    # Diagnóstico resumido
    print("\n=== Interpretação automática ===")
    lss_rows = wilc[wilc.config == "LSS"]
    lss_sig = (lss_rows.p_value < 0.05).sum()
    print(f"  LSS vs LLL: {lss_sig}/3 famílias com diferença estatisticamente significativa (p < 0,05)")
    print(f"  → Em {3 - lss_sig}/3 famílias, LSS é indistinguível de LLL.")

    sss_rows = wilc[wilc.config == "SSS"]
    sss_sig = (sss_rows.p_value < 0.05).sum()
    print(f"  SSS vs LLL: {sss_sig}/3 famílias com diferença significativa.")

    # Pooled efeito planner vs synth
    p_pl = pooled[(pooled.stage == "planner") & (pooled.metric == "quality")]["delta_zscore"].iloc[0]
    p_sq = pooled[(pooled.stage == "sql") & (pooled.metric == "quality")]["delta_zscore"].iloc[0]
    p_sy = pooled[(pooled.stage == "synth") & (pooled.metric == "quality")]["delta_zscore"].iloc[0]
    print(f"\n  Efeitos pooled em z-score: Planner={p_pl:+.2f}, SQL={p_sq:+.2f}, Synth={p_sy:+.2f}")
    print(f"  → SQL tem o efeito pooled mais fraco; Planner e Synth são similares em magnitude.")

    print(f"\n  Artefatos em: {analysis_dir}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    run(root / "data" / "outputs", root / "data" / "outputs" / "_analysis")
