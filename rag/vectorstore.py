import os
from langchain_chroma import Chroma
from rag.config import CHROMA_DIR
from rag.embeddings import get_embeddings

def build_store(topic, chunks):
    path = os.path.join(CHROMA_DIR, topic)
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=path,
        collection_name=topic,
    )

def get_store(topic):
    path = os.path.join(CHROMA_DIR, topic)
    return Chroma(
        persist_directory=path,
        embedding_function=get_embeddings(),
        collection_name=topic,
    )
