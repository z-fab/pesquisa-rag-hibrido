"""Juízes LLM-as-judge para avaliação de respostas.

Fornece três funções:
- judge_sql_result: avalia se resultado SQL é suficiente para responder a pergunta.
- judge_final_result: avalia qualidade da resposta final em 3 dimensões.
- rejudge_snapshots: re-julgamento em lote idempotente, usado para triangulação
  com múltiplos juízes sobre snapshots existentes.

As funções individuais aceitam `llm=None` — se não passado, usam o juiz padrão
do SETTINGS (`JUDGE_PROVIDER`/`JUDGE_MODEL`). Passar `llm` permite triangulação
com múltiplos juízes sem re-executar o pipeline de inferência.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from src.config.providers import get_llm
from src.config.settings import SETTINGS
from src.utils.tracking import parse_llm_json


class SQLJudgeVerdict(BaseModel):
    match: bool = Field(..., description="Whether the SQL result matches expected")
    reasoning: str = Field(..., description="Brief evaluation summary")


class ResponseJudgeVerdict(BaseModel):
    completude: int = Field(..., ge=0, le=2)
    fidelidade: int = Field(..., ge=0, le=2)
    rastreabilidade: int = Field(..., ge=0, le=2)
    reasoning: str = Field(..., description="Brief justification for scores")


def _default_judge_llm() -> BaseChatModel:
    """Retorna o juiz padrão (J1) configurado em SETTINGS."""
    return get_llm(SETTINGS.JUDGE_PROVIDER, SETTINGS.JUDGE_MODEL, SETTINGS)


def _invoke_judge_with_fallback(llm: BaseChatModel, schema, messages) -> dict:
    """Invoca juiz tentando structured output primeiro, caindo em plain + parse manual.

    Alguns provedores (DeepSeek V3.2 via OpenRouter) às vezes emitem control
    chars literais (\\n, \\t) dentro de strings JSON, o que quebra o parser
    estrito do `with_structured_output`. O fallback usa o parser tolerante
    `parse_llm_json` (com `strict=False`).
    """
    # Caminho 1: structured output
    try:
        wrapped = llm.with_structured_output(schema)
        response = wrapped.invoke(messages)
        return response.model_dump()
    except Exception as e:
        logger.debug(f"Structured output falhou ({type(e).__name__}); tentando plain + parse_llm_json")

    # Caminho 2: plain invoke + parse tolerante
    response = llm.invoke(messages)
    parsed = parse_llm_json(response)
    if parsed is None:
        raise ValueError(f"Plain invoke também não produziu JSON parseável. Raw: {getattr(response, 'content', str(response))[:300]}")
    return schema.model_validate(parsed).model_dump()


def judge_sql_result(output: dict, input_data: dict, llm: BaseChatModel | None = None) -> dict:
    """Avalia se o resultado SQL obtido é suficiente para responder a pergunta.

    Args:
        output: dict com `sql_query` e `result_raw` da execução.
        input_data: dict com `question`, `sql_query`, `sql_result` (gabarito).
        llm: LLM a usar. Se None, usa o juiz padrão.
    """
    llm = llm or _default_judge_llm()

    system = """You are a technical evaluator of SQL query results in a scientific experiment.

    Given the <original_question>, decide if the <obtained_result> contains sufficient data
    to answer the question, using <expected_result> as a reference.

    The key criterion is SEMANTIC SUFFICIENCY: does the obtained result contain the data
    needed to answer the question correctly?

    <acceptable_differences>
    - Extra columns beyond those in the expected result (additional data is fine)
    - More rows than expected, as long as the key data is present
    - Column names with the same meaning but different naming
    - Numerical rounding or formatting differences
    - Different row ordering
    </acceptable_differences>

    <unacceptable_differences>
    - Key data absent (e.g., asked for top 5 but only returned 3)
    - Numerical values with significant divergence (>5% on key metrics)
    - Wrong data universe (incorrect year/state/crop filter)
    - Result that would lead to a fundamentally different answer
    </unacceptable_differences>

    Respond with JSON: {"match": true/false, "reasoning": "<brief text>"}"""

    user = f"""
    <original_question>
    {input_data.get("question", "")}
    </original_question>

    <expected_result>
    QUERY: {input_data.get("sql_query", "")}
    RESULT: {json.dumps(input_data.get("sql_result", ""), ensure_ascii=False)}
    </expected_result>

    <obtained_result>
    QUERY: {output.get("sql_query", "")}
    RESULT: {output.get("result_raw", "")}
    </obtained_result>"""

    return _invoke_judge_with_fallback(
        llm, SQLJudgeVerdict,
        [SystemMessage(content=system), HumanMessage(content=user)],
    )


def judge_final_result(output: dict, input_data: dict, llm: BaseChatModel | None = None) -> dict:
    """Avalia a resposta final em completude, fidelidade e rastreabilidade.

    Args:
        output: dict com `final_answer`, `sql_results`, `text_results`.
        input_data: dict com `expected_answer`.
        llm: LLM a usar. Se None, usa o juiz padrão.
    """
    llm = llm or _default_judge_llm()

    system = """You are an answer evaluator in a scientific experiment with LLM agents.

    Evaluate the <obtained_answer> against the <expected_answer> using the <available_sources>
    as ground truth for grounding verification.

    <criteria>
    1) completude (0-2): Does the answer cover all main points of the expected answer?
       - 0: misses most key points
       - 1: covers some but not all key points
       - 2: covers all key points of the expected answer
       IMPORTANT: Additional correct information beyond the expected answer does NOT penalize.
       If the answer covers all expected points AND adds extra relevant details, score 2.

    2) fidelidade (0-2): Is all information grounded in the provided sources? No hallucinations?
       - 0: contains significant fabricated information
       - 1: mostly grounded but some claims lack source support
       - 2: all factual claims are supported by the provided sources

    3) rastreabilidade (0-2): Does it explicitly cite sources (tables, reports, documents)?
       - 0: no source citations at all
       - 1: some citations but incomplete
       - 2: consistently cites sources for key claims
    </criteria>

    Respond with JSON: {"completude": int, "fidelidade": int, "rastreabilidade": int, "reasoning": "<brief text>"}"""

    sql_results = output.get("sql_results", [])
    text_results = output.get("text_results", [])

    sql_context = ""
    for r in sql_results:
        sql_context += f"SQL Query: {r.get('sql_query', '')}\nResult: {r.get('result_raw', '')}\n"

    text_context = ""
    for r in text_results:
        sources = r.get("sources", [])
        text_context += f"Sources: {sources}\n"

    user = f"""
    <expected_answer>
    {input_data.get("expected_answer", "")}
    </expected_answer>

    <available_sources>
    {sql_context}
    {text_context}
    </available_sources>

    <obtained_answer>
    {output.get("final_answer", "")}
    </obtained_answer>"""

    result = _invoke_judge_with_fallback(
        llm, ResponseJudgeVerdict,
        [SystemMessage(content=system), HumanMessage(content=user)],
    )
    result["avg_score"] = round((result["completude"] + result["fidelidade"] + result["rastreabilidade"]) / 3, 2)
    return result


def _configured_judges() -> dict[str, BaseChatModel]:
    """Retorna dict {judge_label: llm} com os juízes configurados em SETTINGS.

    Inclui J1 sempre. Inclui J2/J3 apenas se os PROVIDER/MODEL estiverem setados.
    """
    judges: dict[str, BaseChatModel] = {
        "j1": get_llm(SETTINGS.JUDGE_PROVIDER, SETTINGS.JUDGE_MODEL, SETTINGS),
    }
    if SETTINGS.JUDGE_PROVIDER_2 and SETTINGS.JUDGE_MODEL_2:
        judges["j2"] = get_llm(SETTINGS.JUDGE_PROVIDER_2, SETTINGS.JUDGE_MODEL_2, SETTINGS)
    if SETTINGS.JUDGE_PROVIDER_3 and SETTINGS.JUDGE_MODEL_3:
        judges["j3"] = get_llm(SETTINGS.JUDGE_PROVIDER_3, SETTINGS.JUDGE_MODEL_3, SETTINGS)
    return judges


def _aggregate_judgements(judgements: dict[str, dict]) -> dict:
    """Calcula médias agregadas entre juízes disponíveis.

    Para cada dimensão da síntese e para sql.match, retorna média entre juízes
    que avaliaram aquele item.
    """
    if not judgements:
        return {}

    agg = {"sql": {}, "response": {}}
    n_judges = len(judgements)

    sql_matches = [j["sql"]["match"] for j in judgements.values() if j.get("sql", {}).get("match") is not None]
    if sql_matches:
        agg["sql"]["match_fraction"] = sum(sql_matches) / len(sql_matches)
        agg["sql"]["match_majority"] = sum(sql_matches) > len(sql_matches) / 2
        agg["sql"]["n_judges"] = len(sql_matches)

    for dim in ("completude", "fidelidade", "rastreabilidade"):
        vals = [j["response"].get(dim) for j in judgements.values() if j.get("response", {}).get(dim) is not None]
        if vals:
            agg["response"][dim] = sum(vals) / len(vals)

    if agg["response"]:
        agg["response"]["avg_score"] = (
            agg["response"].get("completude", 0)
            + agg["response"].get("fidelidade", 0)
            + agg["response"].get("rastreabilidade", 0)
        ) / 3
        agg["response"]["n_judges"] = n_judges

    return agg


def rejudge_snapshots(
    outputs_dir: Path | None = None,
    dry_run: bool = False,
    migrate_only: bool = False,
    snapshot_paths: list[Path] | None = None,
    limit: int | None = None,
    concurrency: int = 1,
) -> dict[str, int]:
    """Re-julga snapshots usando os juízes configurados.

    Idempotente: verifica se cada item já foi julgado por cada juiz e pula os existentes.

    Estrutura canônica após execução:
        snapshot[i]["evaluation"]["judgements"] = {
            "j1": {"sql": {...}, "response": {...}},
            "j2": {"sql": {...}, "response": {...}},
            "j3": {"sql": {...}, "response": {...}},
        }
        snapshot[i]["evaluation"]["judgement_agg"] = {...médias...}

    A chave legada `evaluation["judgement"]` é removida durante a migração.

    Args:
        outputs_dir: diretório raiz com subpastas de modelos. Se None, use `snapshot_paths`.
        dry_run: se True, não chama LLMs e não grava arquivos — apenas conta.
        migrate_only: se True, apenas migra o formato legado (sem chamar J2/J3).
        snapshot_paths: lista específica de snapshots a processar (ignora outputs_dir).
        limit: se definido, processa apenas os primeiros N snapshots (útil para teste).

    Returns:
        Dict com contagens: {"items_judged": int, "llm_calls": int, "snapshots_updated": int}.
    """
    if migrate_only:
        judges: dict[str, BaseChatModel] = {}
        logger.info("Modo migrate-only: apenas normalizando formato, sem chamadas LLM")
    else:
        judges = _configured_judges()
        logger.info(f"Re-julgamento com juízes: {list(judges.keys())}")

    stats = {"items_judged": 0, "llm_calls": 0, "snapshots_updated": 0}

    if snapshot_paths is None:
        if outputs_dir is None:
            raise ValueError("precisa passar `outputs_dir` ou `snapshot_paths`")
        snapshot_paths = sorted(outputs_dir.rglob("snapshot_*.json"))
        logger.info(f"Encontrados {len(snapshot_paths)} snapshots em {outputs_dir}")
    else:
        snapshot_paths = sorted(snapshot_paths)
        logger.info(f"Processando {len(snapshot_paths)} snapshots específicos")

    if limit is not None:
        snapshot_paths = snapshot_paths[:limit]
        logger.info(f"Limitado a {len(snapshot_paths)} snapshots")

    # Processa snapshots sequencialmente (um arquivo por vez); itens dentro do arquivo
    # podem rodar em paralelo se `concurrency > 1` (usando asyncio + Semaphore).
    asyncio.run(_rejudge_async(snapshot_paths, judges, dry_run, concurrency, stats))
    return stats


async def _rejudge_async(
    snapshot_paths: list[Path],
    judges: dict[str, BaseChatModel],
    dry_run: bool,
    concurrency: int,
    stats: dict[str, int],
) -> None:
    """Implementação async do re-julgamento, com concorrência via Semaphore."""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    for snap_path in snapshot_paths:
        with open(snap_path, encoding="utf-8") as f:
            snapshots = json.load(f)

        # Lock per-file: serializa writes (vários items podem terminar ao mesmo tempo
        # e cada um dispararia _atomic_write_json; a rename seria race condition).
        write_lock = asyncio.Lock()

        tasks = [
            _process_item_async(
                snap_path=snap_path,
                snapshots=snapshots,
                idx=i,
                judges=judges,
                dry_run=dry_run,
                semaphore=semaphore,
                write_lock=write_lock,
                stats=stats,
            )
            for i in range(len(snapshots))
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Se alguma task re-levantou (ex.: API key exausta), propaga
        for r in results:
            if isinstance(r, Exception):
                raise r

        file_updated = any(results)
        if file_updated and not dry_run:
            stats["snapshots_updated"] += 1
            logger.info(f"Snapshot atualizado: {snap_path.name}")

            # Atualiza o results_*.json correspondente (uma vez por arquivo)
            results_path = snap_path.parent / snap_path.name.replace("snapshot_", "results_", 1)
            if results_path.exists():
                _propagate_to_results(results_path, snapshots)


async def _call_judge_with_retry(fn, *args, max_retries: int = 3, base_delay: float = 1.0):
    """Chama um juiz síncrono em thread com retry exponencial.

    Lida com erros transitórios comuns: respostas vazias/truncadas (ValidationError
    do pydantic quando o LLM retorna `{\\n` por timeout), rate limits, network errors.

    Delays: 1s, 2s, 4s (total até ~7s antes da falha final).
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return await asyncio.to_thread(fn, *args)
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Tentativa {attempt + 1}/{max_retries} falhou ({type(e).__name__}): {str(e)[:150]}. Aguardando {delay}s...")
                await asyncio.sleep(delay)
    raise last_exc


async def _process_item_async(
    snap_path: Path,
    snapshots: list[dict],
    idx: int,
    judges: dict[str, BaseChatModel],
    dry_run: bool,
    semaphore: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    stats: dict[str, int],
) -> bool:
    """Processa um item do snapshot: migra formato + chama juízes faltantes.

    Retorna True se algo foi atualizado.
    """
    async with semaphore:
        snap = snapshots[idx]
        inp = snap["input"]
        out = snap["output"]
        ev = snap.get("evaluation", {})
        item_updated = False

        # Migração do formato legado
        existing = ev.get("judgements") or {}
        if ev.get("judgement") and not existing:
            existing["j1"] = ev["judgement"]
            ev["judgements"] = existing
            item_updated = True
        if "judgement" in ev and existing:
            ev.pop("judgement", None)
            item_updated = True

        # Chama juízes faltantes — um por vez, gravando após cada um.
        # Permite retomada exata se a API falhar no meio.
        for label, llm in judges.items():
            if label in existing:
                continue
            stats["llm_calls"] += 2
            logger.debug(f"Julgando {snap_path.name}[{idx}] com {label}")
            try:
                sql_j = await _call_judge_with_retry(
                    judge_sql_result, _extract_sql_output(out), inp, llm
                )
                resp_j = await _call_judge_with_retry(
                    judge_final_result, out, inp, llm
                )
            except Exception as e:
                logger.error(f"Falha {label} em {snap_path.name}[{idx}] após retries: {e}")
                stats.setdefault("failed_items", 0)
                stats["failed_items"] += 1
                # Grava o que já foi feito e TENTA OS PRÓXIMOS juízes — a falha
                # de 1 juiz não deve impedir j2/j3 de julgarem o mesmo item.
                if item_updated and not dry_run:
                    async with write_lock:
                        ev["judgement_agg"] = _aggregate_judgements(existing)
                        snap["evaluation"] = ev
                        await asyncio.to_thread(_atomic_write_json, snap_path, snapshots)
                continue

            existing[label] = {"sql": sql_j, "response": resp_j}
            item_updated = True

            # Checkpoint incremental
            if not dry_run:
                async with write_lock:
                    ev["judgement_agg"] = _aggregate_judgements(existing)
                    snap["evaluation"] = ev
                    await asyncio.to_thread(_atomic_write_json, snap_path, snapshots)

        # Caso de migração-só (sem juízes novos): grava uma vez no final
        if item_updated:
            ev["judgement_agg"] = _aggregate_judgements(existing)
            snap["evaluation"] = ev
            stats["items_judged"] += 1
            if not dry_run:
                async with write_lock:
                    await asyncio.to_thread(_atomic_write_json, snap_path, snapshots)

        return item_updated


def check_judgement_status(outputs_dir: Path) -> list[dict]:
    """Verifica status de julgamento em todos os snapshots.

    Para cada snapshot, retorna dict com total de itens e contagem por juiz.

    Returns:
        Lista de dicts: [{
            "path": Path,
            "model": str, "arch": str,
            "total": int,
            "by_judge": {"j1": int, "j2": int, "j3": int},
            "complete": int,  # itens com todos os juízes configurados
            "missing_per_judge": {"j1": [ids], "j2": [ids], "j3": [ids]},
        }]
    """
    configured = set(_configured_judges().keys())

    rows = []
    for snap_path in sorted(outputs_dir.rglob("snapshot_*.json")):
        with open(snap_path, encoding="utf-8") as f:
            snaps = json.load(f)

        by_judge = {"j1": 0, "j2": 0, "j3": 0}
        missing = {"j1": [], "j2": [], "j3": []}
        complete = 0

        for s in snaps:
            ev = s.get("evaluation", {})
            judgements = ev.get("judgements", {})
            item_id = s["input"]["id"]
            for j in ("j1", "j2", "j3"):
                if j in judgements:
                    by_judge[j] += 1
                elif j in configured:
                    missing[j].append(item_id)
            if all(j in judgements for j in configured):
                complete += 1

        # Extrair modelo e arquitetura do path (pasta = modelo, nome do arquivo = arch)
        model = snap_path.parent.name
        arch = "?"
        for a in ("full", "no-verifier", "no-synthesizer", "poc"):
            if f"_{a}_" in snap_path.name:
                arch = a
                break

        rows.append({
            "path": snap_path,
            "model": model,
            "arch": arch,
            "total": len(snaps),
            "by_judge": by_judge,
            "complete": complete,
            "missing_per_judge": missing,
        })
    return rows


def _atomic_write_json(path: Path, data) -> None:
    """Escrita atômica: grava em .tmp e renomeia.

    Previne corrupção do arquivo se o processo for interrompido no meio da
    gravação (SIGINT, OOM, energia). O rename é operação atômica no POSIX.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=3, default=str)
    tmp.replace(path)


def _extract_sql_output(output: dict) -> dict:
    """Extrai (sql_query, result_raw) do primeiro SQL executado, no formato esperado por judge_sql_result."""
    sql_results = output.get("sql_results", [])
    executed = [r for r in sql_results if isinstance(r, dict) and r.get("executed")]
    if executed:
        return {"sql_query": executed[0].get("sql_query", ""), "result_raw": executed[0].get("result_raw", "")}
    return {"sql_query": "", "result_raw": ""}


def _propagate_to_results(results_path: Path, snapshots: list[dict]) -> None:
    """Propaga os julgamentos agregados do snapshot para o results_*.json correspondente."""
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    by_id = {s["input"]["id"]: s["evaluation"] for s in snapshots}
    for r in data.get("results", []):
        eid = r.get("id")
        if eid in by_id:
            ev = by_id[eid]
            # Remove chave antiga — estrutura canônica é judgements + judgement_agg
            r.pop("judgement", None)
            r["judgements"] = ev.get("judgements", {})
            r["judgement_agg"] = ev.get("judgement_agg", {})

    _atomic_write_json(results_path, data)
