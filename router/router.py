from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.wellness_agent.agent import agent as wellness_agent
from agents.academic_agent.agent import agent as academic_agent
from agents.career_agent.agent import build_career_agent
from agents.startup_agent.agent import build_graph

load_dotenv()
api = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    api_key=api,
)
startup_agent=build_graph()
career_agent = build_career_agent()

def classify_intent(query: str) -> str:
    prompt = f"""
    Classify the user query into exactly ONE category.

    Categories:
    - wellness  : stress, mental health, sleep, counselling, wellbeing, activities
    - career    : placement, job eligibility, company, role, interview preparation
    - academic  : study topics, syllabus, previous year questions, quiz, study plan
    - startup   : business ideas, entrepreneurship, funding, incubation, startup roadmap, business model, market research, MCA, MSME
    User Query:
    {query}

    Return only one word: wellness, career, startup or academic.
    """
    label = llm.invoke(prompt).content[0]["text"]
    if "wellness" in label:
        return "wellness"
    if "career" in label:
        return "career"
    if "startup" in label:
        return "startup"
    return "academic"

def run_startup(query: str) -> str:

    result = startup_agent.invoke(
        {
        "user_query": query,
        "category": "",
        "priority": "",
        "search_results": "",
        "feasibility": "",
        "recommendation": ""
        }
        )

    return result["recommendation"]

def run_wellness(query: str) -> str:
    response = wellness_agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    msg = response["messages"][-1].content
    return msg if isinstance(msg, str) else msg[0]["text"]

def run_academic(query: str) -> str:
    response = academic_agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    msg = response["messages"][-1].content
    return msg if isinstance(msg, str) else msg[0]["text"]

def run_career(query: str) -> str:
    result = career_agent.invoke({
        "user_input": query,
        "company": "",
        "desired_role": "",
        "cgpa": "",
        "skills": [],
        "experience": "",
        "profile_summary": "",
        "rag_context": "",
        "eligibility_status": "",
        "missing_requirements": [],
        "final_answer": ""
    })
    return result["final_answer"]

def route_query(query: str) -> str:
    intent = classify_intent(query)
    print(f"[Router] Intent detected: {intent}")
    if intent == "wellness":
        return run_wellness(query)
    if intent == "career":
        return run_career(query)
    if intent=="startup":
        return run_startup(query)
    return run_academic(query)

if __name__ == "__main__":
    query =input("Enter a question: ")
    print(route_query(query))