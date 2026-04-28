import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from graph_architecture import app
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# --- CONFIGURAÇÕES ---
INPUT_DATASET = Path("data/validate.json")
RESULTS_FILE = Path("data/resultados_poc.json")
METRICS_FILE = Path("data/metricas_poc.json")
ANSWER_SIM_THRESHOLD = 0.65  # heurística simples de similaridade textual
JUDGE_MODEL = ChatOpenAI(model="gpt-5", temperature=0)
QUALITY_MODEL = ChatOpenAI(model="gpt-5", temperature=0)
SQL_JUDGE_MODEL = ChatOpenAI(model="gpt-5", temperature=0)


class JudgeVerdict(BaseModel):
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Pontuação de similaridade entre 0 e 1",
    )
    match: bool = Field(..., description="Se a resposta está correta/adequada")
    reasoning: str = Field(..., description="Resumo curto da avaliação")


class QualityVerdict(BaseModel):
    completude: int = Field(..., ge=0, le=2)
    fidelidade: int = Field(..., ge=0, le=2)
    rastreabilidade: int = Field(..., ge=0, le=2)
    reasoning: str = Field(..., description="Resumo curto justificando as notas")


class SQLJudgeVerdict(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    match: bool = Field(..., description="Se o resultado SQL responde corretamente ao gabarito")
    reasoning: str = Field(..., description="Resumo da avaliação")


def load_dataset() -> List[dict]:
    if not INPUT_DATASET.exists():
        print(f"Erro: Arquivo {INPUT_DATASET} não encontrado.")
        return []

    with INPUT_DATASET.open("r", encoding="utf-8") as f:
        return json.load(f)


def expected_sources(example: dict) -> Set[str]:
    """Deriva a fonte esperada com base no campo 'type' do dataset."""
    mapping = {
        "S": {"sqldb"},
        "NS": {"vectorstore"},
        "H": {"sqldb", "vectorstore"},
    }
    return mapping.get(example.get("type", "").upper(), set())


def router_label(decision: Optional[str]) -> str:
    mapping = {"sqldb": "S", "vectorstore": "NS", "hybrid": "H"}
    if not decision:
        return "UNK"
    return mapping.get(decision.lower(), decision.upper())


def judge_similarity(question: str, expected: str, answer: str) -> Tuple[float, bool, str]:
    """Usa LLM como juiz para avaliar similaridade da resposta."""
    if not expected or not answer:
        return 0.0, False, "Esperado ou resposta ausentes."

    system = """Você é um avaliador. Compare a resposta do sistema com a resposta esperada.
    Retorne uma nota entre 0 e 1 (0=totalmente incorreta, 1=perfeitamente alinhada).
    A nota deve refletir fidelidade factual e cobertura dos pontos principais.
    """
    user = f"""Pergunta: {question}

Resposta esperada (gabarito):
{expected}

Resposta do sistema:
{answer}
"""
    verdict_reason = ""
    score = 0.0
    match = False
    try:
        structured_model = JUDGE_MODEL.with_structured_output(JudgeVerdict)
        verdict: JudgeVerdict = structured_model.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        score = round(verdict.score, 3)
        match = verdict.match or score >= ANSWER_SIM_THRESHOLD
        verdict_reason = verdict.reasoning
    except Exception as e:
        verdict_reason = f"Falha no juiz LLM: {e}"

    if score == 0.0 and not match:
        # Fallback simples para não zerar em caso de falha do juiz LLM
        simple_score = SequenceMatcher(None, expected.lower(), answer.lower()).ratio()
        score = round(simple_score, 3)
        match = score >= ANSWER_SIM_THRESHOLD
        verdict_reason += " | Fallback SequenceMatcher aplicado."

    return score, match, verdict_reason


def judge_quality(question: str, expected: str, answer: str) -> Tuple[int, int, int, float, str]:
    """Avalia completude, fidelidade e rastreabilidade em escala 0-2 usando LLM."""
    if not expected or not answer:
        return 0, 0, 0, 0.0, "Esperado ou resposta ausentes."

    system = """Você é avaliador de respostas. Atribua notas 0, 1 ou 2 para:
    - completude: cobre todos os pontos?
    - fidelidade: sem alucinações? Apenas informações suportadas pelo gabarito.
    - rastreabilidade: cita as fontes ou dados que suportam a resposta?
    Responda apenas com JSON: {"completude": int, "fidelidade": int, "rastreabilidade": int, "reasoning": "..."}"""
    user = f"""Pergunta: {question}

Resposta esperada (gabarito):
{expected}

Resposta do sistema:
{answer}
"""
    try:
        structured_model = QUALITY_MODEL.with_structured_output(QualityVerdict)
        verdict: QualityVerdict = structured_model.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        avg_score = round(
            (verdict.completude + verdict.fidelidade + verdict.rastreabilidade) / 3, 3
        )
        return (
            verdict.completude,
            verdict.fidelidade,
            verdict.rastreabilidade,
            avg_score,
            verdict.reasoning,
        )
    except Exception as e:
        return 0, 0, 0, 0.0, f"Falha no juiz LLM: {e}"


def overlap_score(expected_docs: List[str], retrieved_docs: List[str]):
    """Retorna a fração de documentos esperados que foram recuperados (0-1)."""
    if not expected_docs:
        return None
    exp = set(expected_docs)
    ret = set(retrieved_docs or [])
    if not exp:
        return None
    return round(len(exp & ret) / len(exp), 3)


def sources_from_trace(trace: List[dict]) -> Set[str]:
    """Mapeia os nós executados para as fontes de dados efetivamente usadas."""
    nodes = {t["node"] for t in trace}
    used: Set[str] = set()
    if "sql_agent" in nodes:
        used.add("sqldb")
    if "text_retriever" in nodes:
        used.add("vectorstore")
    return used


def sql_result_matches(actual: Optional[List[dict]], expected) -> bool:
    if expected is None:
        return False
    if actual is None:
        return False
    # Normaliza expected para lista de dicts
    if isinstance(expected, dict):
        expected_list = [expected]
    elif isinstance(expected, list):
        expected_list = expected
    else:
        return False

    def normalize_rows(rows):
        normed = []
        for row in rows:
            if isinstance(row, dict):
                norm = {}
                for k, v in row.items():
                    if isinstance(v, float):
                        norm[k] = round(v, 6)
                    else:
                        norm[k] = v
                normed.append(norm)
        return normed

    a_norm = normalize_rows(actual)
    e_norm = normalize_rows(expected_list)
    if not a_norm or not e_norm:
        return False

    try:
        return sorted(a_norm, key=lambda x: sorted(x.items())) == sorted(
            e_norm, key=lambda x: sorted(x.items())
        )
    except Exception:
        return False


def judge_sql_result(question: str, expected_sql, actual_sql) -> Tuple[float, bool, str]:
    """Usa LLM como juiz para comparar o resultado SQL com o gabarito de forma flexível."""
    if expected_sql is None or actual_sql is None:
        return 0.0, False, "Resultado esperado ou obtido ausente."

    def to_json(data):
        try:
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            return str(data)

    system = """Você é um avaliador técnico de resultados SQL.
Compare o resultado obtido com o gabarito e decida se responde corretamente à pergunta.
Considere equivalente mesmo que haja colunas extras ou nomes levemente diferentes,
desde que os valores e o conteúdo respondam ao que foi solicitado.
Responda apenas com JSON: {"score": 0-1, "match": true/false, "reasoning": "..."}"""

    user = f"""Pergunta: {question}

Resultado esperado (gabarito):
{to_json(expected_sql)}

Resultado obtido (SQL):
{to_json(actual_sql)}
"""
    try:
        structured = SQL_JUDGE_MODEL.with_structured_output(SQLJudgeVerdict)
        verdict: SQLJudgeVerdict = structured.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        return round(verdict.score, 3), verdict.match, verdict.reasoning
    except Exception as e:
        # Fallback para comparação exata se o juiz falhar
        fallback = sql_result_matches(actual_sql, expected_sql)
        reason = f"Falha no juiz SQL LLM: {e}. Fallback comparação exata={fallback}"
        return (1.0 if fallback else 0.0), fallback, reason


def precision_recall(expected_docs: List[str], retrieved_docs: List[str]) -> Tuple[Optional[float], Optional[float]]:
    if not expected_docs:
        return None, None
    exp = set(expected_docs)
    ret = set(retrieved_docs or [])
    if not ret:
        return 0.0, 0.0
    tp = len(exp & ret)
    precision = tp / len(ret) if ret else 0.0
    recall = tp / len(exp) if exp else None
    return round(precision, 3), round(recall, 3)


def sql_execution_success(sql_result: Optional[str]) -> bool:
    if sql_result is None:
        return False
    if isinstance(sql_result, str) and sql_result.lower().startswith("erro sql"):
        return False
    return True


def calculate_metrics(results: List[dict]) -> Dict[str, object]:
    """Gera métricas agregadas alinhadas com o plano de avaliação."""
    labels = ["S", "NS", "H"]
    confusion = {true: {pred: 0 for pred in labels} for true in labels}

    metrics: Dict[str, object] = {
        "total_questions": len(results),
        "total_errors": 0,
        "routing": {"accuracy": 0.0, "confusion_matrix": confusion},
        "retrieval": {
            "S": {"execution_accuracy": 0.0, "answer_accuracy": 0.0},
            "NS": {"precision": 0.0, "recall": 0.0},
            "H": {"execution_accuracy": 0.0, "answer_accuracy": 0.0, "precision": 0.0, "recall": 0.0},
        },
        "final_answer_quality": {
            "overall_avg": {"completude": 0.0, "fidelidade": 0.0, "rastreabilidade": 0.0, "media": 0.0},
            "by_type": {
                "S": {"completude": 0.0, "fidelidade": 0.0, "rastreabilidade": 0.0, "media": 0.0, "count": 0},
                "NS": {"completude": 0.0, "fidelidade": 0.0, "rastreabilidade": 0.0, "media": 0.0, "count": 0},
                "H": {"completude": 0.0, "fidelidade": 0.0, "rastreabilidade": 0.0, "media": 0.0, "count": 0},
            },
        },
        "efficiency": {
            "avg_total_latency": 0.0,
            "avg_agent_latency": {},
            "avg_token_usage": {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0},
        },
    }

    # Acumuladores
    valid = 0
    routing_hits = 0
    total_latency = 0.0
    agent_totals: Dict[str, float] = {}
    agent_counts: Dict[str, int] = {}
    token_sum = {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0}

    struct_exec_count = 0
    struct_exec_success = 0
    struct_answer_count = 0
    struct_answer_success = 0

    ns_precision_sum = 0.0
    ns_recall_sum = 0.0
    ns_precision_count = 0
    ns_recall_count = 0

    h_exec_count = 0
    h_exec_success = 0
    h_answer_count = 0
    h_answer_success = 0
    h_precision_sum = 0.0
    h_recall_sum = 0.0
    h_precision_count = 0
    h_recall_count = 0

    quality_overall = {"completude": 0.0, "fidelidade": 0.0, "rastreabilidade": 0.0, "media": 0.0}
    quality_counts = 0

    for res in results:
        if res.get("error"):
            metrics["total_errors"] += 1
            continue

        valid += 1
        q_type = res.get("type")
        pred = res.get("router_predicted")

        if q_type in labels and pred in labels:
            metrics["routing"]["confusion_matrix"][q_type][pred] += 1
        if res.get("routing_correct"):
            routing_hits += 1

        # Latência
        lat = res.get("latency_total") or 0.0
        total_latency += lat
        for node, dur in res.get("agent_latencies", {}).items():
            agent_totals[node] = agent_totals.get(node, 0.0) + dur
            agent_counts[node] = agent_counts.get(node, 0) + 1

        # Tokens
        tu = res.get("token_usage") or {}
        token_sum["input_tokens"] += tu.get("input_tokens", 0) or 0
        token_sum["output_tokens"] += tu.get("output_tokens", 0) or 0
        token_sum["total_tokens"] += tu.get("total_tokens", 0) or 0

        # Estruturado
        if q_type == "S":
            struct_exec_count += 1
            if res.get("sql_execution_success"):
                struct_exec_success += 1
            struct_answer_count += 1
            if res.get("sql_result_match"):
                struct_answer_success += 1

        # Híbrido também computa execução separada
        if q_type == "H":
            h_exec_count += 1
            if res.get("sql_execution_success"):
                h_exec_success += 1
            h_answer_count += 1
            if res.get("sql_result_match"):
                h_answer_success += 1

        # Não estruturado / textos
        if q_type == "NS":
            p, r = res.get("precision"), res.get("recall")
            if p is not None:
                ns_precision_sum += p
                ns_precision_count += 1
            if r is not None:
                ns_recall_sum += r
                ns_recall_count += 1
        elif q_type == "H":
            p, r = res.get("precision"), res.get("recall")
            if p is not None:
                h_precision_sum += p
                h_precision_count += 1
            if r is not None:
                h_recall_sum += r
                h_recall_count += 1

        # Qualidade final (0-2)
        if res.get("quality_scores"):
            qs = res["quality_scores"]
            quality_overall["completude"] += qs.get("completude", 0)
            quality_overall["fidelidade"] += qs.get("fidelidade", 0)
            quality_overall["rastreabilidade"] += qs.get("rastreabilidade", 0)
            quality_overall["media"] += res.get("quality_avg", 0)
            quality_counts += 1

            if q_type in metrics["final_answer_quality"]["by_type"]:
                bucket = metrics["final_answer_quality"]["by_type"][q_type]
                bucket["completude"] += qs.get("completude", 0)
                bucket["fidelidade"] += qs.get("fidelidade", 0)
                bucket["rastreabilidade"] += qs.get("rastreabilidade", 0)
                bucket["media"] += res.get("quality_avg", 0)
                bucket["count"] += 1

    if valid > 0:
        metrics["routing"]["accuracy"] = round((routing_hits / valid) * 100, 2)

        metrics["efficiency"]["avg_total_latency"] = round(total_latency / valid, 3)
        metrics["efficiency"]["avg_agent_latency"] = {
            node: round(agent_totals[node] / agent_counts[node], 3)
            for node in agent_totals
            if agent_counts[node] > 0
        }
        metrics["efficiency"]["avg_token_usage"] = {
            "input_tokens": round(token_sum["input_tokens"] / valid, 3),
            "output_tokens": round(token_sum["output_tokens"] / valid, 3),
            "total_tokens": round(token_sum["total_tokens"] / valid, 3),
        }

        # Retrieval estruturado
        if struct_exec_count > 0:
            metrics["retrieval"]["S"]["execution_accuracy"] = round(
                struct_exec_success / struct_exec_count * 100, 2
            )
        if struct_answer_count > 0:
            metrics["retrieval"]["S"]["answer_accuracy"] = round(
                struct_answer_success / struct_answer_count * 100, 2
            )

        if h_exec_count > 0:
            metrics["retrieval"]["H"]["execution_accuracy"] = round(
                h_exec_success / h_exec_count * 100, 2
            )
        if h_answer_count > 0:
            metrics["retrieval"]["H"]["answer_accuracy"] = round(
                h_answer_success / h_answer_count * 100, 2
            )

        if ns_precision_count > 0:
            metrics["retrieval"]["NS"]["precision"] = round(
                ns_precision_sum / ns_precision_count * 100, 2
            )
        if ns_recall_count > 0:
            metrics["retrieval"]["NS"]["recall"] = round(ns_recall_sum / ns_recall_count * 100, 2)

        if h_precision_count > 0:
            metrics["retrieval"]["H"]["precision"] = round(
                h_precision_sum / h_precision_count * 100, 2
            )
        if h_recall_count > 0:
            metrics["retrieval"]["H"]["recall"] = round(h_recall_sum / h_recall_count * 100, 2)

        if quality_counts > 0:
            metrics["final_answer_quality"]["overall_avg"] = {
                "completude": round(quality_overall["completude"] / quality_counts, 3),
                "fidelidade": round(quality_overall["fidelidade"] / quality_counts, 3),
                "rastreabilidade": round(quality_overall["rastreabilidade"] / quality_counts, 3),
                "media": round(quality_overall["media"] / quality_counts, 3),
            }
            for t, bucket in metrics["final_answer_quality"]["by_type"].items():
                if bucket["count"] == 0:
                    continue
                bucket["completude"] = round(bucket["completude"] / bucket["count"], 3)
                bucket["fidelidade"] = round(bucket["fidelidade"] / bucket["count"], 3)
                bucket["rastreabilidade"] = round(bucket["rastreabilidade"] / bucket["count"], 3)
                bucket["media"] = round(bucket["media"] / bucket["count"], 3)

    return metrics


def main():
    dataset = load_dataset()
    if not dataset:
        return

    results = []
    print(f"--- Iniciando Avaliação POC ({len(dataset)} perguntas) ---")

    for i, item in enumerate(dataset):
        q_id = item["id"]
        question = item["question"]
        q_type = item.get("type", "NA").upper()
        expected_answer = item.get("expected_answer", "")
        expected_route = expected_sources(item)

        print(f"[{i + 1}/{len(dataset)}] Processando ID {q_id} (type={q_type})...")

        result_entry = {
            "id": q_id,
            "type": q_type,
            "question": question,
            "expected_answer": expected_answer,
            "expected_sources": list(expected_route),
            "router_decision": None,
            "router_predicted": None,
            "executed_agents": [],
            "actual_sources": [],
            "generated_answer": None,
            "sql_query": None,
            "sql_result": None,
            "sql_result_raw": None,
            "text_debug_sources": [],
            "trace": [],
            "agent_latencies": {},
            "latency_total": 0.0,
            "answer_similarity": 0.0,
            "answer_match": False,
            "answer_judge_reason": None,
            "sql_judge_score": None,
            "sql_judge_reason": None,
            "quality_scores": None,
            "quality_avg": 0.0,
            "quality_reason": None,
            "source_overlap": None,
            "route_match": False,
            "routing_correct": False,
            "precision": None,
            "recall": None,
            "sql_execution_success": False,
            "sql_result_match": False,
            "token_usage": {},
            "error": None,
        }

        try:
            output = app.invoke({"question": question})

            trace = output.get("trace", [])
            agent_latencies = {t["node"]: t.get("duration", 0.0) for t in trace}
            actual_sources_used = sources_from_trace(trace)

            result_entry.update(
                {
                    "router_decision": output.get("router_decision"),
                    "router_predicted": router_label(output.get("router_decision")),
                    "generated_answer": output.get("final_answer"),
                    "sql_query": output.get("sql_query"),
                    "sql_result": output.get("sql_result"),
                    "sql_result_raw": output.get("sql_result_raw"),
                    "text_debug_sources": output.get("sources", []),
                    "trace": trace,
                    "executed_agents": output.get("executed_agents", []),
                    "actual_sources": list(actual_sources_used),
                    "agent_latencies": agent_latencies,
                    "latency_total": output.get("total_latency", 0.0),
                    "token_usage": output.get("token_usage", {}),
                }
            )

            # Métricas de acerto de rota
            result_entry["route_match"] = bool(expected_route) and (actual_sources_used == expected_route)
            result_entry["routing_correct"] = result_entry["router_predicted"] == q_type

            # Recuperação
            if "sqldb" in expected_route:
                result_entry["sql_execution_success"] = sql_execution_success(result_entry["sql_result"])
                sql_score, sql_match, sql_reason = judge_sql_result(
                    question, item.get("sql_result"), result_entry["sql_result"]
                )
                result_entry["sql_result_match"] = sql_match
                result_entry["sql_judge_score"] = sql_score
                result_entry["sql_judge_reason"] = sql_reason

            prec, rec = precision_recall(item.get("source_documents", []), result_entry["text_debug_sources"])
            result_entry["precision"] = prec
            result_entry["recall"] = rec

            # Avaliação de resposta (LLM como juiz para correção) e qualidade
            sim, match, reason = judge_similarity(
                question, expected_answer, result_entry["generated_answer"]
            )
            result_entry["answer_similarity"] = sim
            result_entry["answer_match"] = match
            result_entry["answer_judge_reason"] = reason
            result_entry["source_overlap"] = overlap_score(
                item.get("source_documents", []), result_entry["text_debug_sources"]
            )
            comp, fid, rast, q_avg, q_reason = judge_quality(
                question, expected_answer, result_entry["generated_answer"]
            )
            result_entry["quality_scores"] = {
                "completude": comp,
                "fidelidade": fid,
                "rastreabilidade": rast,
            }
            result_entry["quality_avg"] = q_avg
            result_entry["quality_reason"] = q_reason

        except Exception as e:
            print(f"  -> Erro: {e}")
            result_entry["error"] = str(e)

        results.append(result_entry)

    # 1. Salvar Resultados Detalhados (JSON)
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados detalhados salvos em: {RESULTS_FILE}")

    # 2. Calcular e Salvar Métricas Agregadas
    metrics = calculate_metrics(results)
    with METRICS_FILE.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n--- Resumo das Métricas ---")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Métricas salvas em: {METRICS_FILE}")


if __name__ == "__main__":
    main()
