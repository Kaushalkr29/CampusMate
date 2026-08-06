from langchain_google_genai import ChatGoogleGenerativeAI
from rag.retriever import retrieve
from rag.config import GEMINI_API_KEY

def run(state):
    query = state["user_input"]
    context = retrieve("startup", query)
    prompt = (
        "You are the Startup & Innovation Agent. Answer using ONLY the context below.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                 google_api_key=GEMINI_API_KEY, temperature=0)
    state["answer"] = llm.invoke(prompt).content
    return state
