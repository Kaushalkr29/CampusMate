from rag.config import TOPICS
from rag.loader import load_documents
from rag.chunker import split_documents
from rag.vectorstore import build_store

def ingest_topic(topic):
    docs = load_documents(topic)
    if not docs:
        print(f"No documents found for {topic}")
        return
    chunks = split_documents(docs)
    build_store(topic, chunks)
    print(f"{topic}: stored {len(chunks)} chunks")

def ingest_all():
    for topic in TOPICS:
        ingest_topic(topic)

if __name__ == "__main__":
    ingest_all()
