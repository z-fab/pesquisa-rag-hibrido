import json
from collections import defaultdict
from typing import Tuple

from agent.graph import Graph
from agent.state import AgentState
from config.settings import SETTINGS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field
from repositories.evaluation_repository import load_evaluation_json


class SQLJudgeVerdict(BaseModel):
    match: bool = Field(..., description="Se o resultado SQL corresponde ao esperado")
    reasoning: str = Field(..., description="Resumo da avaliação")


class ResponseJudgeVerdict(BaseModel):
    completude: int = Field(..., ge=0, le=2)
    fidelidade: int = Field(..., ge=0, le=2)
    rastreabilidade: int = Field(..., ge=0, le=2)
    reasoning: str = Field(..., description="Resumo curto justificando as notas")


def judge_sql_result(output: dict, input_data: dict) -> dict:
    model = ChatOpenAI(model=SETTINGS.OPENAI_HARD_MODEL, temperature=0)
    model = model.with_structured_output(SQLJudgeVerdict)

    system = """Você é um avaliador técnico de resultados de consultas SQL em um experimento científico.

    Seu objetivo é decidir se o <resultado obtido> é SEMANTICAMENTE EQUIVALENTE ao <resultado esperado> para a mesma pergunta.

    Regras de equivalência (considere EQUIVALENTE se TODAS forem atendidas):
    1. Diferenças aceitáveis:
    - Ordenação diferente das linhas (a menos que a ordenação seja essencial para a resposta).
    - Nomes de colunas diferentes mas com o mesmo significado.
    - Colunas extras que não alteram a interpretação da resposta.
    - Pequenas diferenças de arredondamento em valores numéricos.
    2. Diferenças NÃO aceitáveis:
    - Linhas faltando ou sobrando em relação ao resultado esperado de forma significativa.
    - Valores numéricos principais diferentes (além de arredondamento).
    - Retornar apenas um subconjunto muito limitado (por exemplo, TOP 1) quando o esperado é um ranking mais completo.
    - Filtros adicionais que restringem o universo de análise de forma relevante (por exemplo, filtrar por um estado específico quando o esperado é o Brasil todo).

    Instruções:
    - Use o <resultado esperado> como fonte de verdade.
    - Compare tabelas linha a linha e coluna a coluna, ignorando apenas diferenças de formatação.
    - Se a resposta obtida traz os dados essenciais para responder a pergunta, considere equivalente.
    - Em "reasoning", explique rapidamente POR QUE os resultados são equivalentes ou não
    (por exemplo: "faltam colunas de área", "filtro de UF alterou o universo", "apenas ordenação diferente").

    Responda SOMENTE com JSON válido neste formato:
    {"match": true/false, "reasoning": "<texto curto em português>"}"""

    user = f"""
    <resultado esperado>
    QUERY: {input_data.get("sql_query", "")}\n
    RESULTADO: {json.dumps(input_data.get("sql_result", ""))}
    </resultado esperado>

    <resultado obtido>
    QUERY: {output.get("sql_query", "")}\n
    RESULTADO: {output.get("sql_result", "")}
    </resultado obtido>
    """

    response = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return response.dict()


def judge_final_result(output: dict, input_data: dict) -> dict:
    model = ChatOpenAI(model=SETTINGS.OPENAI_HARD_MODEL, temperature=0)
    model = model.with_structured_output(ResponseJudgeVerdict)

    system = """Você é avaliador de respostas em um experimento científico de agentes com LLM.

    Você deve avaliar a <resposta obtida> comparando com a <resposta esperada> para completude,
    mas avaliar fidelidade em relação às <fontes disponíveis> (SQL results, RAG documents).

    Atribua notas 0, 1 ou 2 para:

    1) completude:
    2 = cobre todos os pontos principais da resposta esperada (pode reorganizar ou resumir, mas nada importante fica de fora).
    1 = cobre parte relevante, mas omite algum ponto importante ou responde só parcialmente.
    0 = não responde o que foi perguntado, ou ignora a maior parte do que é esperado.

    2) fidelidade:
    Avalie tanto em relação às FONTES (SQL results, RAG docs) quanto à RESPOSTA ESPERADA:
    2 = todas as informações na resposta estão presentes nas fontes E não contém erros factuais
        em relação à resposta esperada; não inventa números ou fatos novos.
    1 = contém pequenas extrapolações ou detalhes adicionais não presentes nas fontes,
        mas que são plausíveis e não contraditórios; OU um erro menor em relação ao esperado.
    0 = contém erros claros (números trocados, afirma "não há dados" quando as fontes trazem dados,
        inventa fatos não presentes nas fontes, ou contradiz significativamente o esperado).

    3) rastreabilidade:
    2 = menciona explicitamente as fontes ou evidências (por exemplo: tabelas, anos, documentos, relatórios)
        de forma que um leitor consiga rastrear de onde vieram as informações.
    1 = faz referência genérica a "dados", "relatórios" ou "texto", sem identificar claramente as fontes,
        ou cita apenas parte das fontes relevantes.
    0 = não menciona nenhuma fonte ou evidência; a resposta parece uma opinião sem referência.

    Importante:
    - Não penalize mudanças de estilo ou reformulação de frases. Foque no conteúdo.
    - Para FIDELIDADE: avalie TANTO se a resposta é fiel às FONTES (SQL/RAG) quanto se não contradiz a resposta esperada.
    Uma resposta pode ser fiel às fontes mas conter erros se as fontes estiverem incompletas ou incorretas.
    - Para COMPLETUDE: compare com a resposta esperada.
    - Explique em "reasoning" de forma curta por que deu essas notas.

    Responda SOMENTE com JSON válido:
    {"completude": int, "fidelidade": int, "rastreabilidade": int, "reasoning": "<texto curto em português>"}"""

    user = f"""
    <resposta esperada>
    RESPOSTA: {input_data.get("expected_answer", "")}
    </resposta esperada>

    <fontes disponíveis>
    SQL Query Executada: {output.get("sql_query", "")}
    SQL Result: {json.dumps(output.get("sql_result", ""))}
    RAG Sources: {output.get("sources", [])}
    </fontes disponíveis>

    <resposta obtida>
    RESPOSTA: {output.get("final_answer", "")}
    </resposta obtida>
    """

    response = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    response = response.dict()
    response["avg_score"] = round(
        (response["completude"] + response["fidelidade"] + response["rastreabilidade"])
        / 3,
        2,
    )

    return response


def evaluate_output(output: dict, item: dict) -> dict:
    def precision_recall(
        expected_docs: list[str], retrieved_docs: list[str]
    ) -> Tuple[float | None, float | None]:
        if not expected_docs:
            return None, None
        exp = set(expected_docs)
        ret = set(retrieved_docs or [])
        if not ret:
            return 0.0, 0.0
        tp = len(exp & ret)
        precision = tp / len(ret) if ret else 0.0
        recall = tp / len(exp) if exp else None
        return round(precision, 2), round(recall, 2)

    result_entry = {}

    # Basic info
    result_entry["id"] = item.get("id", "UNK")
    result_entry["input"] = item.get("question", "")
    result_entry["type"] = item.get("type", "UNK").upper()

    # Expected outputs
    result_entry["expected_answer"] = item.get("expected_answer", "")
    result_entry["expected_agents"] = (
        ["sql_agent"]
        if result_entry["type"] == "S"
        else ["rag_agent"]
        if result_entry["type"] == "NS"
        else ["sql_agent", "rag_agent"]
    )

    # Actual outputs
    result_entry["output_answer"] = output.get("final_answer", "")
    result_entry["output_executed_agents"] = output.get("executed_agents", [])
    # Determine if expected agents were used
    result_entry["output_match_agents"] = all(
        agent in result_entry["output_executed_agents"]
        for agent in result_entry["expected_agents"]
    )
    # Determine output type based on router decision
    result_entry["output_type_predicted"] = (
        "NS"
        if output.get("router_decision", "") == "non_structured"
        else "S"
        if output.get("router_decision", "") == "structured"
        else "H"
    )
    # Determine if output type matches expected type
    result_entry["output_match_type"] = (
        result_entry["output_type_predicted"] == result_entry["type"]
    )

    # SQL Agent Output
    result_entry["output_sql"] = {}
    result_entry["output_sql"]["executed"] = output.get("sql_executed", False)
    result_entry["output_sql"]["query"] = output.get("sql_query", "")
    result_entry["output_sql"]["execution_result"] = output.get("sql_result", "")

    # RAG Agent Output
    result_entry["output_rag"] = {}
    result_entry["output_rag"]["sources"] = output.get("sources", [])

    source_documents = [
        chave
        for documento in item.get("source_documents", [])
        for chave in documento.keys()
    ]
    prec, rec = precision_recall(source_documents, output.get("sources", []))
    result_entry["output_rag"]["precision"] = prec
    result_entry["output_rag"]["recall"] = rec

    # Trace, Latency and Token Usage
    result_entry["output_trace"] = output.get("trace", [])
    result_entry["output_latency"] = {}
    result_entry["output_latency"]["per_agent"] = {
        t["node"]: t.get("duration", 0.0) for t in output.get("trace", [])
    }
    result_entry["output_latency"]["total"] = output.get("total_latency", 0.0)
    result_entry["output_token_usage"] = output.get("token_usage", {})

    # Judgement output
    result_entry["judgement"] = {}
    if result_entry["output_sql"]["executed"]:
        result_entry["judgement"]["sql"] = judge_sql_result(output, item)
    else:
        result_entry["judgement"]["sql"] = {
            "match": False,
            "reasoning": "SQL agent was not executed.",
        }

    result_entry["judgement"]["response"] = judge_final_result(output, item)

    return result_entry


def run_evaluation() -> dict:
    logger.info("Starting evaluation process.")

    dataset = load_evaluation_json()
    if not dataset:
        logger.warning("No evaluation data found.")
        return {}

    results = []
    snapshot = []
    for i, item in enumerate(dataset):
        logger.info(f"Evaluating item {i + 1}/{len(dataset)}.")

        graph = Graph(
            state=AgentState, start_state={"question": item.get("question", "")}
        )
        output = graph.run()

        result_entry = evaluate_output(output, item)
        results.append(result_entry)
        snapshot.append({"input": item, "output": output, "evaluation": result_entry})

    return results, snapshot


def run_evaluation_from_snapshot(snapshots: dict) -> dict:
    logger.info("Starting evaluation from snapshot.")

    results = []
    for i, snap in enumerate(snapshots):
        logger.info(f"Evaluating item {i + 1}/{len(snapshots)}.")
        result_entry = evaluate_output(snap.get("output", {}), snap.get("input", {}))
        results.append(result_entry)

    return results


def calculate_metrics(evaluation_results: list) -> dict:
    # Initialize confusion matrix
    labels = ["S", "NS", "H"]
    confusion = {true: {pred: 0 for pred in labels} for true in labels}

    # Initialize metrics structure
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
                "S": {
                    "completude": 0.0,
                    "fidelidade": 0.0,
                    "rastreabilidade": 0.0,
                    "media": 0.0,
                    "count": 0,
                },
                "NS": {
                    "completude": 0.0,
                    "fidelidade": 0.0,
                    "rastreabilidade": 0.0,
                    "media": 0.0,
                    "count": 0,
                },
                "H": {
                    "completude": 0.0,
                    "fidelidade": 0.0,
                    "rastreabilidade": 0.0,
                    "media": 0.0,
                    "count": 0,
                },
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

    # Accumulate counts for various metrics
    acc_dict = defaultdict(int)
    acc_dict["agent_total_latency"] = defaultdict(float)
    acc_dict["agent_execution_count"] = defaultdict(int)
    acc_dict["quality_overall"] = {
        "completude": 0.0,
        "fidelidade": 0.0,
        "rastreabilidade": 0.0,
        "media": 0.0,
    }

    for res in evaluation_results:
        # Routing metrics
        metrics["routing"]["confusion_matrix"][res["type"]][
            res["output_type_predicted"]
        ] += 1
        acc_dict["routing_hits"] += int(res["output_match_type"])

        # Latency metrics
        acc_dict["total_latency"] += res.get("output_latency", {}).get("total", 0.0)
        for agent, lat in res.get("output_latency", {}).get("per_agent", {}).items():
            acc_dict["agent_total_latency"][agent] += lat
            acc_dict["agent_execution_count"][agent] += 1

        # Token usage metrics
        tu = res.get("output_token_usage", {})
        acc_dict["input_tokens"] += tu.get("input_tokens", 0)
        acc_dict["output_tokens"] += tu.get("output_tokens", 0)
        acc_dict["total_tokens"] += tu.get("total_tokens", 0)

        # Structured (S) retrieval metrics
        if res["type"] == "S":
            acc_dict["S_total"] += 1
            if res["output_sql"]["executed"]:
                acc_dict["S_executed"] += 1
                if res["judgement"]["sql"]["match"]:
                    acc_dict["S_match"] += 1

        # Non-Structured (NS) retrieval metrics
        if res["type"] == "NS":
            acc_dict["NS_total"] += 1
            prec = res["output_rag"].get("precision")
            rec = res["output_rag"].get("recall")
            if prec is not None:
                acc_dict["NS_precision_sum"] += prec
                acc_dict["NS_precision_count"] += 1
            if rec is not None:
                acc_dict["NS_recall_sum"] += rec
                acc_dict["NS_recall_count"] += 1

        # Hybrid (H) retrieval metrics
        if res["type"] == "H":
            acc_dict["H_total"] += 1
            if res["output_sql"]["executed"]:
                acc_dict["H_sql_executed"] += 1
                if res["judgement"]["sql"]["match"]:
                    acc_dict["H_sql_match"] += 1

            prec = res["output_rag"].get("precision")
            rec = res["output_rag"].get("recall")
            if prec is not None:
                acc_dict["H_rag_precision_sum"] += prec
                acc_dict["H_rag_precision_count"] += 1
            if rec is not None:
                acc_dict["H_rag_recall_sum"] += rec
                acc_dict["H_rag_recall_count"] += 1

        # Final answer quality metrics
        rj = res.get("judgement", {}).get("response", {})
        acc_dict["quality_overall"]["completude"] += rj.get("completude", 0)
        acc_dict["quality_overall"]["fidelidade"] += rj.get("fidelidade", 0)
        acc_dict["quality_overall"]["rastreabilidade"] += rj.get("rastreabilidade", 0)
        acc_dict["quality_overall"]["media"] += rj.get("avg_score", 0)
        acc_dict["quality_count"] += 1

        metrics["final_answer_quality"]["by_type"][res["type"]]["completude"] += rj.get(
            "completude", 0
        )
        metrics["final_answer_quality"]["by_type"][res["type"]]["fidelidade"] += rj.get(
            "fidelidade", 0
        )
        metrics["final_answer_quality"]["by_type"][res["type"]]["rastreabilidade"] += (
            rj.get("rastreabilidade", 0)
        )
        metrics["final_answer_quality"]["by_type"][res["type"]]["media"] += rj.get(
            "avg_score", 0
        )
        metrics["final_answer_quality"]["by_type"][res["type"]]["count"] += 1

    # Calculate final metrics
    metrics["routing"]["accuracy"] = round(
        (acc_dict["routing_hits"] / metrics["total_questions"]) * 100, 2
    )

    # Efficiency metrics
    metrics["efficiency"]["avg_total_latency"] = round(
        acc_dict["total_latency"] / metrics["total_questions"], 2
    )
    metrics["efficiency"]["avg_agent_latency"] = {
        agent: round(total_lat / acc_dict["agent_execution_count"][agent], 2)
        for agent, total_lat in acc_dict["agent_total_latency"].items()
    }
    metrics["efficiency"]["avg_token_usage"] = {
        "input_tokens": round(acc_dict["input_tokens"] / metrics["total_questions"]),
        "output_tokens": round(acc_dict["output_tokens"] / metrics["total_questions"]),
        "total_tokens": round(acc_dict["total_tokens"] / metrics["total_questions"]),
    }

    # Structured (S) retrieval metrics
    if acc_dict["S_total"] > 0:
        metrics["retrieval"]["S"]["execution_accuracy"] = round(
            (acc_dict["S_executed"] / acc_dict["S_total"]) * 100, 2
        )
        metrics["retrieval"]["S"]["answer_accuracy"] = (
            round((acc_dict["S_match"] / acc_dict["S_executed"]) * 100, 2)
            if acc_dict["S_executed"] > 0
            else 0.0
        )

    # Non-Structured (NS) retrieval metrics
    if acc_dict["NS_total"] > 0:
        metrics["retrieval"]["NS"]["precision"] = round(
            acc_dict["NS_precision_sum"] / acc_dict["NS_precision_count"], 2
        )
        metrics["retrieval"]["NS"]["recall"] = round(
            acc_dict["NS_recall_sum"] / acc_dict["NS_recall_count"], 2
        )

    # Hybrid (H) retrieval metrics
    if acc_dict["H_total"] > 0:
        metrics["retrieval"]["H"]["execution_accuracy"] = round(
            (acc_dict["H_sql_executed"] / acc_dict["H_total"]) * 100, 2
        )
        metrics["retrieval"]["H"]["answer_accuracy"] = (
            round((acc_dict["H_sql_match"] / acc_dict["H_sql_executed"]) * 100, 2)
            if acc_dict["H_sql_executed"] > 0
            else 0.0
        )
        metrics["retrieval"]["H"]["precision"] = round(
            acc_dict["H_rag_precision_sum"] / acc_dict["H_rag_precision_count"], 2
        )
        metrics["retrieval"]["H"]["recall"] = round(
            acc_dict["H_rag_recall_sum"] / acc_dict["H_rag_recall_count"], 2
        )

    # Final answer quality metrics
    metrics["final_answer_quality"]["overall_avg"] = {
        "completude": round(
            acc_dict["quality_overall"]["completude"] / acc_dict["quality_count"], 2
        ),
        "fidelidade": round(
            acc_dict["quality_overall"]["fidelidade"] / acc_dict["quality_count"], 2
        ),
        "rastreabilidade": round(
            acc_dict["quality_overall"]["rastreabilidade"] / acc_dict["quality_count"],
            2,
        ),
        "media": round(
            acc_dict["quality_overall"]["media"] / acc_dict["quality_count"], 2
        ),
    }

    for t, bucket in metrics["final_answer_quality"]["by_type"].items():
        count = bucket["count"]
        if count > 0:
            bucket["completude"] = round(bucket["completude"] / count, 2)
            bucket["fidelidade"] = round(bucket["fidelidade"] / count, 2)
            bucket["rastreabilidade"] = round(bucket["rastreabilidade"] / count, 2)
            bucket["media"] = round(bucket["media"] / count, 2)

    return metrics
