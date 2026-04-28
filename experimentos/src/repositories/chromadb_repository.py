from src.db.chromadb import get_vectorstore


def similarity_search(query: str, k: int = 5, filter_dict: dict | None = None) -> list[dict]:
    """Performs similarity search in the ChromaDB collection.

    Returns all document metadata alongside content.
    """
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=k, filter=filter_dict)
    return [{"content": doc.page_content, **doc.metadata} for doc in results]
