import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from eval.judges import judge_final_result, judge_sql_result
from eval.metrics import calculate_metrics
from src.agent.ablation import AblationMode
from src.agent.graph import run_graph
from src.config.settings import SETTINGS

console = Console()


def _evaluate_output(output: dict, item: dict) -> dict:
    """Evaluates a single graph output against expected data."""

    def precision_recall(expected_docs: list[str], retrieved_docs: list[str]) -> tuple[float | None, float | None]:
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

    entry = {}
    entry["id"] = item.get("id", "UNK")
    entry["input"] = item.get("question", "")
    entry["type"] = item.get("type", "UNK").upper()
    entry["expected_answer"] = item.get("expected_answer", "")

    entry["expected_agents"] = (
        ["sql_planner_executor"]
        if entry["type"] == "S"
        else ["text_retriever"]
        if entry["type"] == "NS"
        else ["sql_planner_executor", "text_retriever"]
    )

    entry["output_answer"] = output.get("final_answer", "")
    entry["output_executed_agents"] = output.get("executed_agents", [])
    entry["output_match_agents"] = all(a in entry["output_executed_agents"] for a in entry["expected_agents"])

    # Determine predicted type from the agents that were actually executed.
    # Usar executed_agents (ao invés do último planner_output) é mais robusto:
    # retries do Verifier podem sobrescrever planner_output com searches=[], mas
    # os agentes que já rodaram refletem a decisão efetiva de roteamento.
    executed = entry["output_executed_agents"]
    ran_sql = "sql_planner_executor" in executed
    ran_text = "text_retriever" in executed

    if ran_sql and ran_text:
        entry["output_type_predicted"] = "H"
    elif ran_sql:
        entry["output_type_predicted"] = "S"
    elif ran_text:
        entry["output_type_predicted"] = "NS"
    else:
        entry["output_type_predicted"] = "UNK"

    entry["output_match_type"] = entry["output_type_predicted"] == entry["type"]

    sql_results = output.get("sql_results", [])
    entry["output_sql_results"] = sql_results

    text_results = output.get("text_results", [])
    all_sources = []
    for tr in text_results:
        if isinstance(tr, dict):
            all_sources.extend(tr.get("sources", []))

    source_documents = [key for doc in item.get("source_documents", []) for key in doc]
    prec, rec = precision_recall(source_documents, all_sources)
    entry["output_rag"] = {
        "sources": all_sources,
        "precision": prec,
        "recall": rec,
    }

    entry["output_trace"] = output.get("trace", [])

    # Compute total latency from total_start if total_latency not set by graph
    total_latency = output.get("total_latency", 0.0)
    if not total_latency and output.get("total_start"):
        total_latency = round(time.perf_counter() - output["total_start"], 2)

    entry["output_latency"] = {
        "per_agent": {t["node"]: t.get("duration", 0.0) for t in output.get("trace", [])},
        "total": total_latency,
    }
    entry["output_token_usage"] = output.get("token_usage", {})

    entry["judgement"] = {}

    executed_sql = [r for r in sql_results if isinstance(r, dict) and r.get("executed")]
    if executed_sql:
        judge_output = {
            "sql_query": executed_sql[0].get("sql_query", ""),
            "result_raw": executed_sql[0].get("result_raw", ""),
        }
        entry["judgement"]["sql"] = judge_sql_result(judge_output, item)
    else:
        entry["judgement"]["sql"] = {"match": False, "reasoning": "SQL was not executed."}

    entry["judgement"]["response"] = judge_final_result(output, item)

    return entry


_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w.-]")


def _sanitize_for_filename(s: str) -> str:
    """Replaces characters that are unsafe in filenames (e.g. '/' in model names like 'meta-llama/llama-3.1-8b')."""
    return _UNSAFE_FILENAME_CHARS.sub("--", s)


def _build_run_id(run_id: str | None = None, provider: str = "", model: str = "", ablation_mode: str = "full") -> str:
    """Builds a run ID from explicit label or auto-generates one."""
    if run_id:
        return run_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = _sanitize_for_filename(model)
    return f"{provider}_{safe_model}_{ablation_mode}_{timestamp}"


def _get_checkpoint_path(run_id: str, output_dir: Path | None = None) -> Path:
    """Returns the checkpoint file path for a given run_id."""
    base = Path(output_dir) if output_dir else SETTINGS.PATH_DATA / "outputs"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"checkpoint_{run_id}.json"


def _load_checkpoint(path: Path) -> dict:
    """Loads checkpoint from disk, or returns empty structure."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"run_id": "", "metadata": {}, "completed": {}, "failed": {}}


def _save_checkpoint_item(
    path: Path,
    run_id: str,
    metadata: dict,
    item_id: str,
    output: dict | None,
    evaluation: dict | None,
    status: str = "completed",
    error: str | None = None,
) -> None:
    """Saves a single evaluated item to the checkpoint file."""
    checkpoint = _load_checkpoint(path)
    checkpoint["run_id"] = run_id
    checkpoint["metadata"] = metadata

    if status == "completed":
        checkpoint["completed"][item_id] = {
            "output": output,
            "evaluation": evaluation,
        }
        checkpoint["failed"].pop(item_id, None)
    elif status == "failed":
        checkpoint["failed"][item_id] = {
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=3, default=str)


def _build_metadata(ablation_mode: AblationMode = AblationMode.FULL) -> dict:
    """Builds metadata dict from current settings."""
    provider = SETTINGS.PROVIDER
    model_map = {
        "openai": SETTINGS.OPENAI_MODEL,
        "gemini": SETTINGS.GEMINI_MODEL,
        "ollama": SETTINGS.OLLAMA_MODEL,
        "groq": SETTINGS.GROQ_MODEL,
        "openrouter": SETTINGS.OPENROUTER_MODEL,
    }
    model = model_map.get(provider, "unknown")

    return {
        "provider": provider,
        "model": model,
        "embedding_provider": SETTINGS.EMBEDDING_PROVIDER,
        "embedding_model": SETTINGS.EMBEDDING_MODEL,
        "judge_provider": SETTINGS.JUDGE_PROVIDER,
        "judge_model": SETTINGS.JUDGE_MODEL,
        "ablation_mode": ablation_mode.value,
        "timestamp": datetime.now().isoformat(),
    }


async def _process_item(
    item: dict,
    semaphore: asyncio.Semaphore,
    checkpoint_path: Path,
    run_id: str,
    metadata: dict,
    progress: Progress,
    task_id: int,
    ablation_mode: AblationMode = AblationMode.FULL,
) -> tuple[dict | None, dict | None]:
    """Processes a single evaluation item with semaphore-based concurrency."""
    item_id = item.get("id", "UNK")
    question = item.get("question", "")

    async with semaphore:
        try:
            logger.debug(f"Processing item {item_id}: {question[:80]}...")
            output = await asyncio.to_thread(run_graph, question, ablation_mode)
            evaluation = await asyncio.to_thread(_evaluate_output, output, item)

            _save_checkpoint_item(
                path=checkpoint_path,
                run_id=run_id,
                metadata=metadata,
                item_id=item_id,
                output=output,
                evaluation=evaluation,
                status="completed",
            )

            progress.advance(task_id)
            logger.debug(f"Completed item {item_id}")
            return evaluation, {"input": item, "output": output, "evaluation": evaluation}

        except Exception as e:
            logger.error(f"Failed item {item_id}: {e}")
            _save_checkpoint_item(
                path=checkpoint_path,
                run_id=run_id,
                metadata=metadata,
                item_id=item_id,
                output=None,
                evaluation=None,
                status="failed",
                error=str(e),
            )
            progress.advance(task_id)
            return None, None


def run_evaluation(
    concurrency: int = 1,
    run_id: str | None = None,
    resume: bool = False,
    ablation_mode: AblationMode = AblationMode.FULL,
    output_dir: Path | None = None,
) -> tuple[list, list, str]:
    """Runs the full evaluation suite with async concurrency and checkpointing."""
    return asyncio.run(_run_evaluation_async(concurrency, run_id, resume, ablation_mode, output_dir))


async def _run_evaluation_async(
    concurrency: int = 1,
    run_id: str | None = None,
    resume: bool = False,
    ablation_mode: AblationMode = AblationMode.FULL,
    output_dir: Path | None = None,
) -> tuple[list, list, str]:
    """Async implementation of the evaluation runner."""
    logger.info("Starting evaluation...")

    metadata = _build_metadata(ablation_mode)
    effective_run_id = _build_run_id(
        run_id, provider=metadata["provider"], model=metadata["model"], ablation_mode=ablation_mode.value
    )
    metadata["run_id"] = effective_run_id

    with open(SETTINGS.PATH_EVAL_FILE, encoding="utf-8") as f:
        dataset = json.load(f)

    if not dataset:
        console.print("[yellow]No evaluation data found.[/yellow]")
        return [], [], effective_run_id

    # Warm-up do vectorstore ChromaDB antes do paralelismo — evita race condition
    # de inicialização quando nodes sql_planner_executor e text_retriever rodam
    # em paralelo dentro do grafo.
    from src.db.chromadb import get_vectorstore
    logger.debug("Warming up ChromaDB vectorstore...")
    get_vectorstore()

    checkpoint_path = _get_checkpoint_path(effective_run_id, output_dir)

    # Load checkpoint for resume
    completed_ids: set[str] = set()
    results: list[dict] = []
    snapshots: list[dict] = []

    if resume:
        checkpoint = _load_checkpoint(checkpoint_path)
        completed_ids = set(checkpoint.get("completed", {}).keys())
        if completed_ids:
            logger.info(f"Resuming run '{effective_run_id}': {len(completed_ids)} items already completed")
            console.print(f"[cyan]Resuming from checkpoint: {len(completed_ids)} items already completed[/cyan]")
            for item_id, data in checkpoint["completed"].items():
                results.append(data["evaluation"])
                snapshots.append(
                    {
                        "input": next((i for i in dataset if i.get("id") == item_id), {}),
                        "output": data["output"],
                        "evaluation": data["evaluation"],
                    }
                )

    # Filter out already-completed items
    pending_items = [item for item in dataset if item.get("id", "UNK") not in completed_ids]

    if not pending_items:
        console.print("[green]All items already completed.[/green]")
        return results, snapshots, effective_run_id

    logger.info(f"Run '{effective_run_id}': {len(pending_items)} items to process (concurrency={concurrency})")

    semaphore = asyncio.Semaphore(concurrency)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Evaluating...", total=len(pending_items))

        tasks = [
            _process_item(
                item, semaphore, checkpoint_path, effective_run_id, metadata, progress, task_id, ablation_mode
            )
            for item in pending_items
        ]

        task_results = await asyncio.gather(*tasks)

    for evaluation, snapshot in task_results:
        if evaluation is not None:
            results.append(evaluation)
            snapshots.append(snapshot)

    metadata["total_questions"] = len(results)
    logger.info(
        f"Evaluation complete: {len(results)} succeeded, "
        f"{len(pending_items) - len([r for r, _ in task_results if r is not None])} failed"
    )

    return results, snapshots, effective_run_id


def save_results(
    results: list,
    snapshots: list,
    run_id: str,
    ablation_mode: AblationMode = AblationMode.FULL,
    output_dir: Path | None = None,
) -> Path:
    """Saves final results and snapshots, cleans up checkpoint."""
    metadata = _build_metadata(ablation_mode)
    metadata["run_id"] = run_id
    metadata["total_questions"] = len(results)

    metrics = calculate_metrics(results)

    base = Path(output_dir) if output_dir else SETTINGS.PATH_DATA / "outputs"
    base.mkdir(parents=True, exist_ok=True)

    results_path = base / f"results_{run_id}.json"
    snapshot_path = base / f"snapshot_{run_id}.json"

    eval_output = {"metadata": metadata, "results": results, "metrics": metrics}

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, ensure_ascii=False, indent=3, default=str)

    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshots, f, ensure_ascii=False, indent=3, default=str)

    # Clean up checkpoint only if all items finished successfully.
    # If any items are missing or failed, keep the checkpoint so --resume can
    # pick up where it left off.
    checkpoint_path = _get_checkpoint_path(run_id, output_dir)
    if checkpoint_path.exists():
        checkpoint = _load_checkpoint(checkpoint_path)
        with open(SETTINGS.PATH_EVAL_FILE, encoding="utf-8") as f:
            dataset_size = len(json.load(f))
        completed_count = len(checkpoint.get("completed", {}))
        failed_count = len(checkpoint.get("failed", {}))
        if completed_count >= dataset_size and failed_count == 0:
            checkpoint_path.unlink()
            logger.debug(f"Removed checkpoint file: {checkpoint_path}")
        else:
            logger.warning(
                f"Keeping checkpoint: {completed_count}/{dataset_size} completed, "
                f"{failed_count} failed. Use --resume to retry."
            )
            console.print(
                f"\n[yellow]⚠ Checkpoint preserved: {completed_count}/{dataset_size} completed"
                + (f", {failed_count} failed" if failed_count else ", some items missing")
                + f". Re-run with --resume --run-id {run_id} to retry.[/yellow]"
            )

    console.print(f"\n[green]Results saved to: {results_path}[/green]")
    console.print(f"[green]Snapshot saved to: {snapshot_path}[/green]")

    return results_path
