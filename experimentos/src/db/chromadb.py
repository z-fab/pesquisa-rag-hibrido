import threading

from langchain_chroma import Chroma

from src.config.providers import get_embeddings
from src.config.settings import SETTINGS

_vectorstore: Chroma | None = None
_init_lock = threading.Lock()


def get_vectorstore() -> Chroma:
    """Retorna o vectorstore ChromaDB singleton, criando sob lock se necessário.

    Thread-safe: usa double-checked locking para evitar que múltiplas threads
    criem instâncias concorrentes do cliente. Sob eval com paralelismo de nós
    (sql_planner_executor e text_retriever rodam em paralelo dentro do grafo
    LangGraph), sem o lock múltiplas inicializações apontariam para o mesmo
    `persist_directory` e corromperiam o estado interno do SQLite do Chroma.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    with _init_lock:
        if _vectorstore is None:
            _vectorstore = Chroma(
                persist_directory=str(SETTINGS.PATH_CHROMA_DB),
                embedding_function=get_embeddings(SETTINGS),
            )
    return _vectorstore
