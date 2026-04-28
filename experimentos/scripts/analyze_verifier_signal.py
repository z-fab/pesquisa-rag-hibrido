"""Análise do Experimento 1 — Verifier signal-only.

Lê o campo `evaluation.verifier_signal_only` dos snapshots `no-verifier` dos 9
modelos principais, extrai as features do sinal e correlaciona com a quality
avaliada pelos 3 juízes. Produz:

    data/outputs/_analysis/verifier_signal_summary.csv
    data/outputs/_analysis/verifier_signal_by_model.csv
    data/outputs/_analysis/verifier_signal_auc.csv
    data/outputs/_analysis/verifier_signal_auc_by_model.csv

Métricas calculadas:
  1. Spearman (com IC 95% bootstrap) entre features Verifier × quality total
     e × cada dimensão (completude, fidelidade, rastreabilidade), pooled e
     por modelo.
  2. AUC-ROC para `overall_pass` e `pct_supported` como classificadores de
     "quality alta" (thresholds 1,3 e 1,5).
  3. Latência adicional: distribuição do verifier_latency.

Uso:
    uv run python -m scripts.analyze_verifier_signal
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computa AUC-ROC via trapezoidal sobre TPR/FPR sweep.

    Implementação sem sklearn. y_true binário {0,1}; y_score contínuo (maior = mais positivo).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        raise ValueError("AUC indefinido: precisa de 2 classes em y_true")

    # Caso degenerado: feature constante → classificador é chance pura.
    # Sem esta checagem, AUC dependeria da ordem da sort e viria enviesada pela prevalência.
    if np.all(y_score == y_score[0]):
        return 0.5

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC indefinido: apenas uma classe")

    # AUC via Mann-Whitney (soma de ranks com desempate médio).
    # Equivalente ao trapezoidal em dados sem ties, mas robusto a ties.
    ranks = scipy_stats.rankdata(y_score)
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)

warnings.filterwarnings("ignore")

MODEL_DISPLAY = {
    "gemini-flash-lite": "Gemini Flash-Lite",
    "gemini-flash": "Gemini Flash",
    "gemini-pro": "Gemini Pro",
    "gpt-nano": "GPT-5 Nano",
    "gpt-mini": "GPT-5 Mini",
    "gpt": "GPT-5",
    "qwen-35b": "Qwen 3.5 35B",
    "qwen-122b": "Qwen 3.5 122B",
    "qwen-397b": "Qwen 3.5 397B",
}

MAIN_MODEL_ORDER = [
    "gemini-flash-lite", "gemini-flash", "gemini-pro",
    "gpt-nano", "gpt-mini", "gpt",
    "qwen-35b", "qwen-122b", "qwen-397b",
]

# Features do sinal do Verifier que vamos correlacionar com quality
FEATURES = [
    "overall_pass",           # bool
    "completeness_covered",   # bool
    "n_missing_aspects",      # int (inverso-correlato)
    "pct_supported",          # 0-1
    "pct_not_supported",      # 0-1 (inverso)
    "pct_partial",            # 0-1 (inverso parcial)
]

DIMENSIONS = ["completude", "fidelidade", "rastreabilidade", "quality"]


def _signal_file_for(snap_path: Path) -> Path:
    """Arquivo paralelo `.verifier_signal.json` ao snapshot."""
    return snap_path.with_name(snap_path.stem + ".verifier_signal.json")


def collect_rows(outputs_dir: Path) -> pd.DataFrame:
    """Lê snapshots no-verifier + arquivo paralelo .verifier_signal.json e extrai
    features + quality avaliada pelos 3 juízes."""
    rows = []
    for snap_path in sorted(outputs_dir.rglob("snapshot_*no-verifier*.json")):
        if "_analysis" in snap_path.parts or "_mix" in snap_path.parts:
            continue
        if snap_path.name.endswith(".verifier_signal.json"):
            continue
        model_key = snap_path.parent.name
        if model_key not in MODEL_DISPLAY:
            continue

        signal_file = _signal_file_for(snap_path)
        if not signal_file.exists():
            continue

        with open(signal_file, encoding="utf-8") as f:
            signals_by_id = json.load(f)

        with open(snap_path, encoding="utf-8") as f:
            items = json.load(f)

        for item in items:
            iid = str(item.get("input", {}).get("id"))
            signal = signals_by_id.get(iid)
            if not signal or "error" in signal:
                continue

            ev = item.get("evaluation", {})
            agg = ev.get("judgement_agg")
            if not agg:
                continue
            resp = agg.get("response", {}) or {}
            c = resp.get("completude")
            f_ = resp.get("fidelidade")
            r = resp.get("rastreabilidade")
            if c is None or f_ is None or r is None:
                continue

            rows.append({
                "model_key": model_key,
                "model": MODEL_DISPLAY[model_key],
                "item_id": iid,
                "type": item.get("input", {}).get("type"),
                # Features Verifier
                "overall_pass": int(signal["overall_pass"]),
                "completeness_covered": int(signal["completeness_covered"]),
                "n_missing_aspects": signal["n_missing_aspects"],
                "pct_supported": signal.get("pct_supported"),
                "pct_not_supported": signal.get("pct_not_supported"),
                "pct_partial": signal.get("pct_partial"),
                "verifier_latency": signal.get("verifier_latency"),
                # Quality (juízes)
                "completude": c,
                "fidelidade": f_,
                "rastreabilidade": r,
                "quality": (c + f_ + r) / 3,
            })
    return pd.DataFrame(rows)


def _bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    """Spearman com IC 95% por bootstrap."""
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    if len(x) < 3:
        return (float("nan"), float("nan"), float("nan"))
    rho = scipy_stats.spearmanr(x, y).correlation
    rng = np.random.default_rng(seed)
    n = len(x)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = scipy_stats.spearmanr(x[idx], y[idx]).correlation
        if not np.isnan(r):
            samples.append(r)
    if not samples:
        return (rho, float("nan"), float("nan"))
    return (rho, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz feature × dimension com Spearman e IC 95%."""
    rows = []
    for feat in FEATURES:
        for dim in DIMENSIONS:
            x = df[feat].to_numpy(dtype=float)
            y = df[dim].to_numpy(dtype=float)
            rho, lo, hi = _bootstrap_spearman_ci(x, y)
            rows.append({
                "feature": feat,
                "dimension": dim,
                "spearman_rho": round(rho, 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "n": int((~(np.isnan(x) | np.isnan(y))).sum()),
            })
    return pd.DataFrame(rows)


def compute_correlations_by_model(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_key, sub in df.groupby("model_key"):
        for feat in FEATURES:
            for dim in DIMENSIONS:
                x = sub[feat].to_numpy(dtype=float)
                y = sub[dim].to_numpy(dtype=float)
                rho, _, _ = _bootstrap_spearman_ci(x, y, n_boot=2000)  # menos boot por célula (9 modelos)
                rows.append({
                    "model_key": model_key,
                    "model": MODEL_DISPLAY[model_key],
                    "feature": feat,
                    "dimension": dim,
                    "spearman_rho": round(rho, 4),
                    "n": len(sub),
                })
    return pd.DataFrame(rows)


def compute_auc_roc(df: pd.DataFrame, quality_threshold: float) -> pd.DataFrame:
    """AUC-ROC de cada feature como classificador binário de quality >= threshold."""
    rows = []
    y_true = (df["quality"] >= quality_threshold).astype(int)
    if y_true.nunique() < 2:
        print(f"  [warn] threshold={quality_threshold}: apenas uma classe — AUC indefinido")
        return pd.DataFrame()
    pooled_prevalence = y_true.mean()

    for feat in FEATURES:
        x = df[feat].to_numpy(dtype=float)
        valid = ~np.isnan(x)
        if valid.sum() < 10:
            continue
        try:
            # Para features invertidas (n_missing, not_supported, partial), usamos -x
            sign = -1 if feat.startswith(("n_missing", "pct_not", "pct_partial")) else 1
            auc = roc_auc_score(y_true[valid], sign * x[valid])
            rows.append({
                "feature": feat,
                "threshold": quality_threshold,
                "auc": round(auc, 4),
                "n": int(valid.sum()),
                "prevalence_positive": round(pooled_prevalence, 3),
            })
        except Exception as e:
            print(f"  [warn] AUC({feat}, thresh={quality_threshold}) falhou: {e}")
    return pd.DataFrame(rows)


def compute_auc_by_model(df: pd.DataFrame, feat: str, quality_threshold: float = 1.3) -> pd.DataFrame:
    """AUC por modelo para uma feature específica."""
    rows = []
    sign = -1 if feat.startswith(("n_missing", "pct_not", "pct_partial")) else 1
    for model_key, sub in df.groupby("model_key"):
        y_true = (sub["quality"] >= quality_threshold).astype(int)
        x = sub[feat].to_numpy(dtype=float)
        valid = ~np.isnan(x)
        if valid.sum() < 10 or y_true.nunique() < 2:
            rows.append({"model_key": model_key, "model": MODEL_DISPLAY[model_key], "auc": None, "n": int(valid.sum())})
            continue
        try:
            auc = roc_auc_score(y_true[valid], sign * x[valid])
            rows.append({
                "model_key": model_key,
                "model": MODEL_DISPLAY[model_key],
                "auc": round(auc, 4),
                "n": int(valid.sum()),
            })
        except Exception:
            rows.append({"model_key": model_key, "model": MODEL_DISPLAY[model_key], "auc": None, "n": int(valid.sum())})
    return pd.DataFrame(rows)


def run(outputs_dir: Path, analysis_dir: Path):
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lendo snapshots no-verifier em {outputs_dir}...")
    df = collect_rows(Path(outputs_dir))

    if df.empty:
        print("\n[ERRO] Nenhum item com verifier_signal_only encontrado.")
        print("       Rode `rag exp-verifier-signal` primeiro.")
        return

    print(f"Itens coletados: {len(df)} (de {df['model_key'].nunique()} modelos)")
    print(f"Distribuição por modelo:")
    print(df.groupby("model_key").size().to_string())

    # Estatísticas descritivas do Verifier signal
    print("\n=== Estatísticas descritivas do sinal ===")
    desc = df[["overall_pass", "completeness_covered", "n_missing_aspects",
               "pct_supported", "verifier_latency"]].describe()
    print(desc.to_string())
    print(f"\n  overall_pass (True): {df['overall_pass'].mean():.1%}")
    print(f"  completeness_covered (True): {df['completeness_covered'].mean():.1%}")
    print(f"  verifier_latency: média {df['verifier_latency'].mean():.2f}s, "
          f"mediana {df['verifier_latency'].median():.2f}s")

    # Correlações pooled
    print("\n=== Correlações pooled (Spearman + IC 95% bootstrap) ===")
    corr_df = compute_correlations(df)
    print(corr_df.to_string(index=False))
    corr_df.to_csv(analysis_dir / "verifier_signal_summary.csv", index=False)

    # Correlações por modelo
    print("\n=== Correlações por modelo ===")
    corr_by_model = compute_correlations_by_model(df)
    corr_by_model.to_csv(analysis_dir / "verifier_signal_by_model.csv", index=False)

    # AUC pooled (threshold 1.3 e 1.5)
    print("\n=== AUC-ROC pooled (threshold quality >= 1,3) ===")
    auc_13 = compute_auc_roc(df, 1.3)
    print(auc_13.to_string(index=False))

    print("\n=== AUC-ROC pooled (threshold quality >= 1,5) ===")
    auc_15 = compute_auc_roc(df, 1.5)
    print(auc_15.to_string(index=False))

    auc_combined = pd.concat([auc_13, auc_15], ignore_index=True)
    auc_combined.to_csv(analysis_dir / "verifier_signal_auc.csv", index=False)

    # AUC por modelo para features principais
    auc_per_model = pd.concat(
        [compute_auc_by_model(df, feat, 1.3).assign(feature=feat)
         for feat in ["pct_supported", "n_missing_aspects", "overall_pass"]],
        ignore_index=True,
    )
    auc_per_model.to_csv(analysis_dir / "verifier_signal_auc_by_model.csv", index=False)

    # Resumo interpretativo — com detecção de casos degenerados
    print("\n=== Interpretação automática ===")

    # Detectar features constantes (variância zero) — caso degenerado em modelos "verifier-silencioso"
    constant_features = [f for f in FEATURES if df[f].nunique(dropna=True) <= 1]
    if constant_features:
        print(f"  ⚠ Features constantes (sem variância): {constant_features}")
        print("    Correlação/AUC indefinidas nessas features — Verifier não discrimina neste regime.")

    # Número de modelos e de valores distintos por feature (diagnóstico de cobertura)
    n_models_covered = df["model_key"].nunique()
    print(f"  Modelos cobertos: {n_models_covered}/9")

    if n_models_covered < 3:
        print(f"\n  ⚠ INCONCLUSIVO — {n_models_covered} modelo(s) só. Precisa de ao menos 3-4 modelos com")
        print("    comportamentos distintos do Verifier para estabelecer correlação confiável.")
        print("    Modelos mais informativos a rodar: GPT-5 Nano, Qwen 35B-A3B, Qwen 122B")
        print("    (Verifier hiperativo) + GPT-5 e Gemini-Flash (verifier intermediário).")
        return

    # Caso real — múltiplos modelos, features com variância
    quality_corrs = corr_df.loc[corr_df["dimension"] == "quality"]
    valid_corrs = quality_corrs.dropna(subset=["spearman_rho"])
    if valid_corrs.empty:
        print("\n  ⚠ INCONCLUSIVO — todas as features são constantes no pool atual.")
        return

    best_feature = valid_corrs.sort_values(
        "spearman_rho", key=lambda s: s.abs(), ascending=False
    ).iloc[0]
    print(f"  Feature com maior |Spearman| vs quality: {best_feature['feature']} "
          f"(ρ = {best_feature['spearman_rho']:.3f}, IC [{best_feature['ci_low']:.2f}; {best_feature['ci_high']:.2f}])")

    if not auc_13.empty:
        # Filtra features degeneradas
        auc_valid = auc_13[~auc_13["feature"].isin(constant_features)]
        if auc_valid.empty:
            print("\n  ⚠ INCONCLUSIVO em AUC — features constantes dominam o pool.")
            return

        best_auc = auc_valid.sort_values("auc", ascending=False).iloc[0]
        print(f"  Feature com maior AUC (threshold 1,3, excluindo constantes): {best_auc['feature']} "
              f"(AUC = {best_auc['auc']:.3f})")

        # Critério de decisão
        if best_auc["auc"] >= 0.75 and abs(best_feature["spearman_rho"]) >= 0.5:
            recommendation = "✓ Sinal útil — ATIVAR no produto como score de triagem padrão"
        elif best_auc["auc"] >= 0.60 or abs(best_feature["spearman_rho"]) >= 0.3:
            recommendation = "⚠ Sinal marginal — uso limitado a triagem humana em casos extremos"
        else:
            recommendation = "✗ Sinal fraco — ABANDONAR; não justifica chamada extra"
        print(f"\n  Recomendação para o produto: {recommendation}")

    print(f"\n[green]Análise concluída.[/green] Artefatos em: {analysis_dir}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    run(root / "data" / "outputs", root / "data" / "outputs" / "_analysis")
