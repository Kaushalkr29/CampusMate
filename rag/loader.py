import os
from langchain_community.document_loaders import (
    DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
)
from rag.config import DATA_DIR

def load_documents(topic):
    folder = os.path.join(DATA_DIR, topic)
    docs = []
    docs += DirectoryLoader(folder, glob="**/*.pdf", loader_cls=PyPDFLoader).load()
    docs += DirectoryLoader(folder, glob="**/*.docx", loader_cls=Docx2txtLoader).load()
    docs += DirectoryLoader(folder, glob="**/*.txt", loader_cls=TextLoader,
                            loader_kwargs={"encoding": "utf-8"}).load()
    for d in docs:
        d.metadata["topic"] = topic
        d.metadata["source_name"] = os.path.basename(d.metadata.get("source", ""))
    return docs
