# CampusMate RAG Module

Creates 4 separate ChromaDB vector stores (syllabus, startup, placement, pyqs)
using Google Gemini embeddings, and lets each agent retrieve context by function.

## Agent -> ChromaDB mapping
- academic_agent  -> syllabus + pyqs
- career_agent    -> placement
- startup_agent   -> startup
- wellness_agent  -> no RAG store (profile/LLM only)

## Setup
1. Put your source files (pdf/docx/txt) into:
   rag/data/syllabus, rag/data/startup, rag/data/placement, rag/data/pyqs
2. Set your API key:
   export GEMINI_API_KEY="your_key"      # Windows: set GEMINI_API_KEY=your_key
3. Build all 4 vector stores (run once):
   python -m rag.ingest

## Install
pip install -r rag/requirements.txt
