"""Mineração qualitativa de snapshots — gera markdown estruturado.

Produz `<analysis_dir>/qualitative_samples.md` com tabelas formatadas que servem
como referência durante a escrita da dissertação. Não substitui interpretação
humana; serve para **encontrar rapidamente** exemplos concretos.

Seções do markdown gerado:
1. Erros de roteamento (match_type=False) por modelo × arquitetura
2. Erros de SQL (sql_match=False) com reasoning do juiz
3. Itens com retry (count > 0) por modelo
4. Comparações lado-a-lado Completa vs Simples nos mesmos itens H
5. Alucinações detectadas (fidelidade=0) com reasoning

Uso via CLI: `rag analyze`
Uso direto: `uv run python -m scripts.qualitative`
"""

from __future__ import annotations

import json
from pathlib import Path

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
    "ministral-3b": "Ministral 3B",
    "gemma3-4b": "Gemma 3 4B",
    "llama-8b": "Llama 3.1 8B",
}

ARCH_DISPLAY = {"full": "Completa", "no-verifier": "Sem Verificação",
                "no-synthesizer": "Sem Síntese", "poc": "Simples"}


def _arch_from_filename(name: str) -> str:
    for a in ARCH_DISPLAY:
        if f"_{a}_" in name:
            return a
    return "unknown"


def _judgement_from_snapshot_item(ev: dict) -> dict:
    """Retorna judgement agregado (novo formato) ou None se faltar.

    Emite warning somente se também não achar formato legado.
    """
    if ev.get("judgement_agg") and ev.get("judgements"):
        return {"agg": ev["judgement_agg"], "judgements": ev["judgements"]}
    return None


def _truncate(s: str, n: int = 200) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def _collect_items(outputs_dir: Path) -> list[dict]:
    """Varre snapshots e retorna lista de dicts prontos para análise qualitativa."""
    items = []
    for snap_path in sorted(outputs_dir.rglob("snapshot_*.json")):
        # Pula arquivos auxiliares de experimento (ex.: .verifier_signal.json)
        if snap_path.name.endswith(".verifier_signal.json"):
            continue
        model_key = snap_path.parent.name
        if model_key not in MODEL_DISPLAY:
            continue
        arch = _arch_from_filename(snap_path.name)
        with open(snap_path, encoding="utf-8") as f:
            snaps = json.load(f)
        if not isinstance(snaps, list):
            continue
        for s in snaps:
            ev = s.get("evaluation", {})
            inp = s.get("input", {})
            out = s.get("output", {})
            jd = _judgement_from_snapshot_item(ev)

            # SQL result extraction
            sql_results = out.get("sql_results", []) or []
            sql_query = sql_results[0].get("sql_query", "") if sql_results else ""

            items.append({
                "model_key": model_key,
                "model": MODEL_DISPLAY[model_key],
                "architecture": arch,
                "arch_display": ARCH_DISPLAY.get(arch, arch),
                "item_id": inp.get("id"),
                "type": inp.get("type"),
                "question": inp.get("question", ""),
                "expected_answer": inp.get("expected_answer", ""),
                "output_answer": out.get("final_answer", ""),
                "sql_query": sql_query,
                "type_predicted": ev.get("output_type_predicted"),
                "match_type": ev.get("output_match_type"),
                "retry_count": out.get("retry_count", 0) or 0,
                "jd": jd,
            })
    return items


def _section_roteamento(items: list[dict]) -> str:
    lines = [
        "## 1. Erros de roteamento (match_type = False)",
        "",
        "Itens onde o roteador classificou incorretamente o tipo da pergunta.",
        "",
        "| Modelo | Arquitetura | Item | Tipo real | Tipo predito |",
        "|---|---|---|---|---|",
    ]
    count = 0
    for it in items:
        if it["match_type"] is False:
            lines.append(f"| {it['model']} | {it['arch_display']} | {it['item_id']} | {it['type']} | {it['type_predicted']} |")
            count += 1
            if count >= 50:
                lines.append("| ... | ... | ... | ... | ... |")
                break
    lines.append("")
    lines.append(f"Total de erros de roteamento listados: {count}.")
    return "\n".join(lines)


def _section_sql_errors(items: list[dict]) -> str:
    lines = [
        "## 2. Erros de SQL (sql_match = False)",
        "",
        "SQL gerada não produz resultado equivalente ao gabarito, conforme juiz agregado.",
        "",
    ]
    count = 0
    for it in items:
        if it["type"] not in ("S", "H"):
            continue
        jd = it.get("jd")
        if not jd:
            continue
        agg_sql = jd["agg"].get("sql", {})
        # Maioria dos juízes marcou como não-match
        if agg_sql.get("match_majority") is False:
            lines.append(f"### {it['model']} · {it['arch_display']} · {it['item_id']} ({it['type']})")
            lines.append(f"**Pergunta:** {_truncate(it['question'], 200)}")
            lines.append(f"**SQL gerada:** `{_truncate(it['sql_query'], 300)}`")
            lines.append("**Reasoning dos juízes:**")
            for label, j in jd["judgements"].items():
                sql_j = j.get("sql", {}) or {}
                r = sql_j.get("reasoning") or "_(sem reasoning)_"
                match = sql_j.get("match")
                match_str = "✓" if match is True else ("✗" if match is False else "?")
                lines.append(f"- **{label}** [{match_str}]: {_truncate(r, 250)}")
            lines.append("")
            count += 1
            if count >= 20:
                lines.append("*(... mais itens omitidos para brevidade ...)*")
                break
    lines.append(f"\nTotal de erros SQL listados: {count}.")
    return "\n".join(lines)


def _section_retries(items: list[dict]) -> str:
    from collections import defaultdict
    counts = defaultdict(lambda: {"items": 0, "retries": 0})
    for it in items:
        if it["architecture"] != "full":
            continue
        key = (it["model_key"], it["model"])
        if it["retry_count"] > 0:
            counts[key]["items"] += 1
            counts[key]["retries"] += it["retry_count"]

    lines = [
        "## 3. Contagem de retries do Verifier (apenas arquitetura Completa)",
        "",
        "| Modelo | Itens com retry | Total de retries |",
        "|---|---|---|",
    ]
    for (_, model), c in sorted(counts.items(), key=lambda x: -x[1]["retries"]):
        lines.append(f"| {model} | {c['items']} | {c['retries']} |")
    if not counts:
        lines.append("| — | — | — |")
    lines.append("")
    return "\n".join(lines)


def _section_full_vs_poc(items: list[dict]) -> str:
    """Compara Completa vs Simples nos mesmos itens H."""
    from collections import defaultdict
    by_model_item = defaultdict(dict)
    for it in items:
        if it["type"] != "H":
            continue
        by_model_item[(it["model_key"], it["item_id"])][it["architecture"]] = it

    lines = [
        "## 4. Comparação Completa vs Simples em itens H",
        "",
        "Amostra de itens híbridos onde as duas arquiteturas produziram respostas diferentes.",
        "",
    ]
    count = 0
    for (mk, item_id), archs in sorted(by_model_item.items()):
        if "full" not in archs or "poc" not in archs:
            continue
        full_it = archs["full"]
        poc_it = archs["poc"]
        # Quality dos agregados
        full_q = _quality_from_jd(full_it.get("jd"))
        poc_q = _quality_from_jd(poc_it.get("jd"))
        if full_q is None or poc_q is None or abs(full_q - poc_q) < 0.5:
            continue  # só mostra onde há diferença notável
        lines.append(f"### {full_it['model']} · {item_id}")
        lines.append(f"**Pergunta:** {_truncate(full_it['question'], 180)}")
        lines.append(f"- **Completa** (quality={full_q:.2f}): {_truncate(full_it['output_answer'], 300)}")
        lines.append(f"- **Simples** (quality={poc_q:.2f}): {_truncate(poc_it['output_answer'], 300)}")
        lines.append("")
        count += 1
        if count >= 15:
            lines.append("*(... mais itens omitidos ...)*")
            break
    lines.append(f"\nTotal de comparações listadas: {count}.")
    return "\n".join(lines)


def _quality_from_jd(jd) -> float | None:
    if not jd:
        return None
    resp = jd.get("agg", {}).get("response", {})
    c = resp.get("completude")
    f = resp.get("fidelidade")
    r = resp.get("rastreabilidade")
    if c is None or f is None or r is None:
        return None
    return (c + f + r) / 3


def _section_alucinacoes(items: list[dict]) -> str:
    lines = [
        "## 5. Alucinações detectadas (fidelidade = 0)",
        "",
        "Itens onde o juiz agregado marcou fidelidade baixa — potenciais alucinações.",
        "",
    ]
    count = 0
    for it in items:
        jd = it.get("jd")
        if not jd:
            continue
        fid = jd.get("agg", {}).get("response", {}).get("fidelidade")
        if fid is None or fid >= 0.5:
            continue
        lines.append(f"### {it['model']} · {it['arch_display']} · {it['item_id']} ({it['type']})")
        lines.append(f"**Pergunta:** {_truncate(it['question'], 180)}")
        lines.append(f"**Resposta:** {_truncate(it['output_answer'], 250)}")
        lines.append(f"**Fidelidade agregada:** {fid:.2f}")
        lines.append("**Reasoning dos juízes:**")
        for label, j in jd["judgements"].items():
            resp = j.get("response", {}) or {}
            r = resp.get("reasoning") or "_(sem reasoning)_"
            jf = resp.get("fidelidade")
            fid_str = f"fid={jf}" if jf is not None else "fid=?"
            lines.append(f"- **{label}** [{fid_str}]: {_truncate(r, 300)}")
        lines.append("")
        count += 1
        if count >= 20:
            lines.append("*(... mais itens omitidos ...)*")
            break
    lines.append(f"\nTotal de alucinações listadas: {count}.")
    return "\n".join(lines)


def run(outputs_dir: Path, analysis_dir: Path) -> Path:
    outputs_dir = Path(outputs_dir)
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    items = _collect_items(outputs_dir)
    n_with_jd = sum(1 for it in items if it["jd"])
    if n_with_jd == 0:
        print("[warn] nenhum item tem `judgements` (formato novo). "
              "Rode `rag rejudge` para migrar o formato legado.")

    md = [
        "# Amostras qualitativas — referência para escrita da dissertação",
        "",
        "Gerado automaticamente por `scripts/qualitative.py`. Este documento NÃO substitui "
        "interpretação humana; serve para encontrar rapidamente exemplos concretos que "
        "ilustrem os padrões observados nos testes estatísticos.",
        "",
        f"Itens analisados: {len(items)} ({n_with_jd} com julgamento no formato novo).",
        "",
        "---",
        "",
        _section_roteamento(items),
        "",
        _section_sql_errors(items),
        "",
        _section_retries(items),
        "",
        _section_full_vs_poc(items),
        "",
        _section_alucinacoes(items),
        "",
    ]

    path = analysis_dir / "qualitative_samples.md"
    path.write_text("\n".join(md), encoding="utf-8")
    print(f"Gravado: {path}")
    return path


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    run(root / "data" / "outputs", root / "data" / "outputs" / "_analysis")
