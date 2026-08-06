from rag.config import TOP_K
from rag.vectorstore import get_store

def retrieve(topic, query, k=TOP_K):
    store = get_store(topic)
    results = store.similarity_search(query, k=k)
    return format_context(results)

def format_context(results):
    blocks = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source_name", "unknown")
        blocks.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(blocks) if blocks else "No relevant context found."
