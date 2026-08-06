from langchain_huggingface import HuggingFaceEmbeddings
from rag.config import EMBEDDING_MODEL, GEMINI_API_KEY

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
