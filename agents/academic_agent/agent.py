from langchain.tools import tool
from langchain.agents import create_agent
from rag.retriever import retrieve
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
api=os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0,api_key=api)

@tool
def explain_topic(query: str) -> str:
    """Explain an academic topic using retrieved textbook context."""
    context = retrieve("syllabus", query)
    prompt = f"""
    Retrieved Context:
    {context}

    Student Question:
    {query}

    Explain the topic simply with Definition, Key Concepts, Examples, Summary.
    """
    return llm.invoke(prompt).content

@tool
def pyq_search(query: str) -> str:
    """Retrieve and summarize relevant previous year questions."""
    context = retrieve("pyqs", query)
    prompt = f"""
    Retrieved Context:
    {context}

    Student Query:
    {query}

    List relevant Previous Year Questions with Question, Year (if available), Topic.
    """
    return llm.invoke(prompt).content

@tool
def generate_quiz(query: str) -> str:
    """Generate MCQ quiz questions from retrieved syllabus context."""
    context = retrieve("syllabus", query)
    prompt = f"""
    Retrieved Context:
    {context}

    Topic:
    {query}

    Generate 5 MCQs with Answers and brief explanations.
    """
    return llm.invoke(prompt).content

@tool
def study_plan(query: str) -> str:
    """Create a study plan using retrieved academic material."""
    context = retrieve("syllabus", query)
    prompt = f"""
    Retrieved Context:
    {context}

    Student Requirement:
    {query}

    Generate a study plan with Topics, Daily schedule, Practice recommendations, Revision strategy.
    """
    return llm.invoke(prompt).content

tools = [explain_topic, pyq_search, generate_quiz, study_plan]

agent = create_agent(
    tools=tools,
    model=llm,
)
if __name__=="__main__":

    response = agent.invoke({"messages": [{"role":"user","content":"Explain python programming"}]})
    print(response["messages"][-1].content[0]["text"])