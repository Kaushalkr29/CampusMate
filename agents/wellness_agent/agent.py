from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from rag.retriever import retrieve
from langchain.agents import create_agent

load_dotenv()
api = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    api_key=api,
)

def get_context(query: str) -> str:
    return retrieve("wellness", query)   # retrieve already returns a formatted string

@tool
def recommend_activity(query: str) -> str:
    """Recommend wellness activities."""
    context = get_context(query)
    prompt = f"""
    Context:
    {context}

    User Query:
    {query}

    Recommend wellness activities.
    """
    return llm.invoke(prompt).content

@tool
def sleep_advice(query: str) -> str:
    """Provide sleep guidance."""
    context = get_context(query)
    prompt = f"""
    Context:
    {context}

    User Query:
    {query}

    Give sleep advice.
    """
    return llm.invoke(prompt).content

@tool
def counselling_support(query: str) -> str:
    """Provide counselling recommendations."""
    context = get_context(query)
    prompt = f"""
    Context:
    {context}

    User Query:
    {query}

    Suggest counselling support.
    """
    return llm.invoke(prompt).content

@tool
def create_wellness_plan(query: str) -> str:
    """Create personalized wellness plan."""
    context = get_context(query)          # fixed: single argument
    prompt = f"""
    Context:
    {context}

    User Query:
    {query}

    Create a personalized wellness plan.
    """
    return llm.invoke(prompt).content

tools = [
    recommend_activity,
    sleep_advice,
    counselling_support,
    create_wellness_plan,
]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a supportive wellness assistant. Use the tools to help the user.",
)
if __name__=="__main__":
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "I am stressed because of my project"}]}
    )

    msg = response["messages"][-1].content
    print(msg if isinstance(msg, str) else msg[0]["text"])