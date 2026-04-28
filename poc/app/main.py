import argparse
import json
from datetime import datetime

from agent.graph import Graph
from agent.state import AgentState
from config.settings import SETTINGS
from loguru import logger
from services.evaluation_service import (
    calculate_metrics,
    run_evaluation,
    run_evaluation_from_snapshot,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sistema v1.0.0")
    parser.add_argument(
        "runner",
        choices=["agent", "validate", "recalculate", "plot"],
        help="Modo de execução do Sistema",
    )

    args = parser.parse_args()

    if args.runner == "agent":
        logger.info("Iniciando o Agent...")

        question = "Qual foi a produção total de açaí no Brasil em 2023 e qual foi a área colhida naquele ano?"

        graph = Graph(state=AgentState, start_state={"question": question})
        output = graph.run()

        logger.info(f"\n✅ Resposta Final:\n{output}")

    elif args.runner == "validate":
        logger.info("Iniciando Validação...")

        result, snapshot = run_evaluation()
        logger.info(f"\n✅ Resultados da Validação:\n{json.dumps(result, indent=3)}")

        result_metrics = calculate_metrics(result)
        logger.info(
            f"\n✅ Métricas da Validação:\n{json.dumps(result_metrics, indent=3)}"
        )

        # Salva resultados em arquivo
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_results_output = (
            SETTINGS.PATH_DATA / "outputs" / f"eval_results_{date_time}.json"
        )
        eval_results = {
            "results": result,
            "metrics": result_metrics,
        }
        with open(file_results_output, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=3)
            logger.info(f"\n✅ Resultados salvos em: {file_results_output}")

        # Salva snapshot em arquivo
        file_snapshot_output = (
            SETTINGS.PATH_DATA / "outputs" / f"eval_snapshot_{date_time}.json"
        )
        with open(file_snapshot_output, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=3)
            logger.info(f"\n✅ Snapshot salvo em: {file_snapshot_output}")

    elif args.runner == "snapshot":
        snapshot_file_name = "eval_results_20251126_171941.json"
        file_input = SETTINGS.PATH_DATA / "outputs" / snapshot_file_name
        logger.info(f"Revalidando a partir de Snapshot: {file_input}")

        with open(file_input, "r", encoding="utf-8") as f:
            snapshots = json.load(f)

        result = run_evaluation_from_snapshot(snapshots)
        result_metrics = calculate_metrics(result)

        logger.info(f"\n✅ Resultados Recalculados:\n{json.dumps(result, indent=3)}")

        logger.info(
            f"\n✅ Métricas Recalculadas:\n{json.dumps(result_metrics, indent=3)}"
        )

        # Salva resultados em arquivo
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_results_output = (
            SETTINGS.PATH_DATA / "outputs" / f"eval_snapshot_results_{date_time}.json"
        )
        eval_results = {
            "results": result,
            "metrics": result_metrics,
        }
        with open(file_results_output, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=3)
            logger.info(
                f"\n✅ Resultados Recalculados salvos em: {file_results_output}"
            )

    elif args.runner == "plot":
        logger.info("Iniciando Plot...")

        graph = Graph(state=AgentState, start_state={"question": ""})
        result = graph.print_graph()

        logger.info(f"\n✅ Graph Plot:\n{result}")
