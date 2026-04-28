from db.chromadb import vectorstore


def similarity_search(
    query: str, k: int = 5, filter_dict: dict | None = None
) -> list[dict]:
    """Performs a similarity search in the ChromaDB collection."""
    results = vectorstore.similarity_search(query, k=k, filter=filter_dict)
    return [
        {"source": doc.metadata.get("source", "unknown"), "content": doc.page_content}
        for doc in results
    ]
