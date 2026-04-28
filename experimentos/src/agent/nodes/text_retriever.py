import time

from loguru import logger

from src.agent.state import AgentState, SearchTask
from src.config.settings import SETTINGS
from src.repositories.chromadb_repository import similarity_search
from src.utils.tracking import record_end


def _execute_single_task(task: SearchTask, k: int) -> dict:
    """Executes a single text search task. Returns a result dict with raw chunk dicts."""
    logger.debug(f"Starting text search for task: {task.query} with sources: {task.sources}")

    search_filter = {"source": {"$in": task.sources}} if task.sources else None
    chunks = similarity_search(query=task.query, k=k, filter_dict=search_filter)

    sources = list({doc.get("source", "unknown") for doc in chunks})

    logger.debug(f"Finished text search: {task.query} — {len(chunks)} chunks from {sources}")

    return {
        "task_query": task.query,
        "chunks": chunks,
        "sources": sources,
    }


def text_retriever_node(state: AgentState) -> dict:
    """Text retriever node: runs similarity search for each text SearchTask."""
    logger.info("Executing Text Retriever...")

    node_start = time.perf_counter()

    tasks = [t for t in state["planner_output"].searches if t.type == "text"]
    all_results = [_execute_single_task(task, SETTINGS.TEXT_SEARCH_K) for task in tasks]

    tracking = record_end("text_retriever", SETTINGS.EMBEDDING_PROVIDER, SETTINGS.EMBEDDING_MODEL, node_start, None)

    return {
        "text_results": all_results,
        **tracking,
    }
