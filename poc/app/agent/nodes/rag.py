import time

from agent.state import AgentState
from loguru import logger
from repositories.chromadb_repository import similarity_search
from utils.agent_utils import record_end


class RAGAgent:
    def __call__(self, state: AgentState):
        logger.info(f"\n🤖 Executando {self.__class__.__name__}...")

        node_start = time.perf_counter()

        selected_docs = state.get("router_docs")
        search_filter = None
        if selected_docs:
            search_filter = {"source": {"$in": selected_docs}}

        logger.debug(f"\n➡️ {search_filter}")

        chunks = similarity_search(
            query=state["question"], k=5, filter_dict=search_filter
        )

        context = "\n---\n".join(
            [f"FONTE: {doc['source']}\nCONTENT: {doc['content']}" for doc in chunks]
        )
        sources = [doc["source"] for doc in chunks]

        tracking_data = record_end(state, "rag_agent", node_start, None)
        logger.debug(f"\n⬅️ {context}")

        return {"text_result": context, "sources": sources, **tracking_data}
