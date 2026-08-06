import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DATA_DIR = "rag/data"
CHROMA_DIR = "rag/chroma_store"

TOPICS = ["syllabus", "startup", "placement", "pyqs","wellness"]

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 4
