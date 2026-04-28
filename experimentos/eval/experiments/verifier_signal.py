"""Experimento 1 — Verifier signal-only (pós-processamento).

Invoca o Verifier UMA única vez sobre respostas já produzidas pela arquitetura
`no-verifier` (snapshots existentes), sem laço de retry. Mede a utilidade do
sinal emitido como indicador de qualidade, sem custo de retrys.

Motivação (ver §4.11.1-a do capítulo de resultados): a Parte 3 mostrou que o
Verifier + retry contribui zero em quality e custa 36 s/item. Mas o *score*
isolado pode ter valor para triagem/auditoria. Este experimento testa se
o sinal correlaciona com quality real (avaliado pelos 3 juízes LLM).

**Persistência**: para cada snapshot `snapshot_X.json`, grava um arquivo
paralelo `snapshot_X.verifier_signal.json` no mesmo diretório. O arquivo
contém `{item_id: features_do_sinal}`. Snapshots originais não são modificados.

**Idempotência**: o arquivo de resultados é lido no início; itens já presentes
são pulados. Parar no meio e reexecutar retoma de onde parou.

**Gravação incremental**: cada item é gravado logo após processado (atômico
via tmp + rename), protegido por lock. Cancelar o processo não perde trabalho
em andamento.

**Paralelismo**: ThreadPoolExecutor — múltiplos itens em paralelo, um Verifier
por thread. Compatível com chamadas LLM bloqueantes (I/O).
"""

from __future__ import annotations

import ast
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from src.agent.nodes.verifier import verifier_node
from src.agent.state import Reference, SynthesizerOutput, SynthesizerSegment


# Parser AST seguro do synthesizer_output legado (str(repr) → Pydantic)

def _node_to_kwargs(call_node: ast.Call) -> dict:
    kwargs = {}
    for kw in call_node.keywords:
        kwargs[kw.arg] = _eval_ast(kw.value)
    return kwargs


def _eval_ast(node):
    """Avaliador AST restrito — aceita literais, listas e chamadas a
    SynthesizerSegment/Reference apenas."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_eval_ast(el) for el in node.elts]
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError(f"Chamada não-nomeada não suportada: {ast.dump(node)}")
        cls_name = node.func.id
        kwargs = _node_to_kwargs(node)
        if cls_name == "SynthesizerSegment":
            return SynthesizerSegment(**kwargs)
        if cls_name == "Reference":
            return Reference(**kwargs)
        raise ValueError(f"Classe não permitida: {cls_name}")
    raise ValueError(f"Nó AST não suportado: {type(node).__name__}")


def _parse_synthesizer_output_repr(so_repr: str) -> SynthesizerOutput:
    """Reconstrói SynthesizerOutput a partir do repr() serializado no snapshot."""
    expr_src = so_repr[len("segments="):] if so_repr.startswith("segments=") else so_repr
    try:
        tree = ast.parse(expr_src, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Sintaxe inválida em synthesizer_output: {e}") from e

    if not isinstance(tree, ast.Expression):
        raise ValueError("AST não é uma Expression")

    segments = _eval_ast(tree.body)
    if not isinstance(segments, list):
        raise ValueError(f"Esperado lista de segments, obtido {type(segments).__name__}")
    return SynthesizerOutput(segments=segments)


def _signal_file_for(snap_path: Path) -> Path:
    """Arquivo paralelo de resultados para um dado snapshot."""
    return snap_path.with_name(snap_path.stem + ".verifier_signal.json")


def _reconstruct_state(item: dict) -> dict:
    out = item["output"]
    so_raw = out.get("synthesizer_output")
    if isinstance(so_raw, str):
        so = _parse_synthesizer_output_repr(so_raw)
    elif isinstance(so_raw, dict):
        so = SynthesizerOutput.model_validate(so_raw)
    else:
        raise ValueError(f"synthesizer_output tipo inesperado: {type(so_raw).__name__}")
    return {
        "question": item["input"]["question"],
        "synthesizer_output": so,
        "sql_results": out.get("sql_results", []),
        "text_results": out.get("text_results", []),
        "retry_count": 0,
    }


def _extract_signal_features(verifier_out, latency: float) -> dict:
    segs = verifier_out.segments
    n = len(segs)
    n_supported = sum(1 for s in segs if s.verdict == "supported")
    n_not_supported = sum(1 for s in segs if s.verdict == "not_supported")
    n_partial = sum(1 for s in segs if s.verdict == "partial")
    return {
        "overall_pass": verifier_out.overall_pass,
        "completeness_covered": verifier_out.completeness.covered,
        "n_missing_aspects": len(verifier_out.completeness.missing_aspects),
        "missing_aspects": verifier_out.completeness.missing_aspects,
        "n_segments": n,
        "n_supported": n_supported,
        "n_not_supported": n_not_supported,
        "n_partial": n_partial,
        "pct_supported": (n_supported / n) if n else None,
        "pct_not_supported": (n_not_supported / n) if n else None,
        "pct_partial": (n_partial / n) if n else None,
        "feedback": verifier_out.feedback,
        "verifier_latency": round(latency, 3),
    }


def _atomic_write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def _load_signal_file(signal_path: Path) -> dict:
    """Lê arquivo de sinais se existir. Formato: {item_id: {...features...}}."""
    if not signal_path.exists():
        return {}
    try:
        with open(signal_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Arquivo de sinais corrompido, recomeçando: {signal_path} ({e})")
        return {}


# Processamento paralelo por snapshot

def _process_item_with_lock(
    item: dict,
    signal_file: Path,
    results: dict,
    lock: threading.Lock,
) -> tuple[str, bool]:
    """Processa 1 item e grava incrementalmente. Retorna (item_id, success)."""
    item_id = str(item["input"].get("id"))
    try:
        state = _reconstruct_state(item)
        t0 = time.perf_counter()
        verifier_out_dict = verifier_node(state)
        latency = time.perf_counter() - t0
        features = _extract_signal_features(verifier_out_dict["verifier_output"], latency)
    except Exception as e:
        features = {"error": str(e)}
        success = False
    else:
        success = True

    # Gravação incremental sob lock
    with lock:
        results[item_id] = features
        _atomic_write_json(signal_file, results)

    return item_id, success


def process_snapshot(
    snap_path: Path,
    concurrency: int = 1,
    force: bool = False,
) -> dict:
    """Processa um snapshot com paralelismo e gravação incremental.

    Retorna {n_total, n_processed, n_skipped, n_failed, signal_file}.
    """
    with open(snap_path, encoding="utf-8") as f:
        snapshot_items = json.load(f)

    signal_file = _signal_file_for(snap_path)
    existing_results = {} if force else _load_signal_file(signal_file)
    lock = threading.Lock()

    # Filtra itens pendentes
    pending_items = []
    skipped = 0
    for item in snapshot_items:
        iid = str(item["input"].get("id"))
        if iid in existing_results and "error" not in existing_results[iid]:
            skipped += 1
            continue
        pending_items.append(item)

    stats = {
        "n_total": len(snapshot_items),
        "n_processed": 0,
        "n_skipped": skipped,
        "n_failed": 0,
        "signal_file": signal_file,
    }

    if not pending_items:
        logger.info(f"  Todos os {stats['n_total']} itens já processados. Nada a fazer.")
        return stats

    logger.info(
        f"  Processando {len(pending_items)} itens (skipped={skipped}), "
        f"concurrency={concurrency}..."
    )

    # Executor paralelo
    if concurrency <= 1:
        for item in pending_items:
            iid, ok = _process_item_with_lock(item, signal_file, existing_results, lock)
            if ok:
                stats["n_processed"] += 1
            else:
                stats["n_failed"] += 1
            logger.info(f"    [{iid}] {'✓' if ok else '✗'}")
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(_process_item_with_lock, item, signal_file, existing_results, lock)
                for item in pending_items
            ]
            for fut in as_completed(futures):
                iid, ok = fut.result()
                if ok:
                    stats["n_processed"] += 1
                else:
                    stats["n_failed"] += 1
                logger.info(f"    [{iid}] {'✓' if ok else '✗'}")

    return stats


def run_on_outputs_dir(
    outputs_dir: Path,
    model_keys: list[str] | None = None,
    concurrency: int = 1,
    force: bool = False,
) -> dict:
    """Processa todos os snapshots `no-verifier` das pastas indicadas.

    Args:
        outputs_dir: raiz de `data/outputs/`.
        model_keys: restringe a pastas específicas (ex.: ["gemini-pro"]). None = todas.
        concurrency: nº de threads paralelas dentro de cada snapshot. Default 1.
        force: se True, reprocessa mesmo itens já processados.

    Retorna dict com totais agregados.
    """
    outputs_dir = Path(outputs_dir)
    totals = {"n_total": 0, "n_processed": 0, "n_skipped": 0, "n_failed": 0, "snapshots": 0}

    for snap in sorted(outputs_dir.rglob("snapshot_*no-verifier*.json")):
        if "_analysis" in snap.parts or "_mix" in snap.parts:
            continue
        model_folder = snap.parent.name
        if model_keys and model_folder not in model_keys:
            continue

        logger.info(f"Processando {snap.relative_to(outputs_dir)}...")
        stats = process_snapshot(snap, concurrency=concurrency, force=force)
        for k in ("n_total", "n_processed", "n_skipped", "n_failed"):
            totals[k] += stats[k]
        totals["snapshots"] += 1
        logger.info(
            f"  → total={stats['n_total']}, processed={stats['n_processed']}, "
            f"skipped={stats['n_skipped']}, failed={stats['n_failed']}, "
            f"salvo em: {stats['signal_file'].name}"
        )

    return totals
