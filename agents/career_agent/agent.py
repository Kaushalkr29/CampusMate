from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from rag.retriever import retrieve
import os
import json
import re

load_dotenv()
api = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    api_key=api
)

class CareerState(TypedDict):
    user_input: str
    company: str
    desired_role: str
    cgpa: str
    skills: List[str]
    experience: str
    profile_summary: str
    rag_context: str
    eligibility_status: str
    missing_requirements: List[str]
    final_answer: str

def get_text(response):
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return content[0].get("text", "")
    return str(content)

def extract_json(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    return {}

def infer_user_profile(state: CareerState) -> CareerState:
    prompt = f"""
Extract career details from the user input.

User Input:
{state["user_input"]}

Infer these fields:
- company
- desired_role
- cgpa
- skills
- experience
- profile_summary

Rules:
- If something is missing, write "not mentioned".
- skills must be a list.
- Return only valid JSON.

Format:
{{
  "company": "",
  "desired_role": "",
  "cgpa": "",
  "skills": [],
  "experience": "",
  "profile_summary": ""
}}
"""
    response = llm.invoke(prompt)
    data = extract_json(get_text(response))
    return {
        **state,
        "company": data.get("company", "not mentioned"),
        "desired_role": data.get("desired_role", "not mentioned"),
        "cgpa": data.get("cgpa", "not mentioned"),
        "skills": data.get("skills", []),
        "experience": data.get("experience", "not mentioned"),
        "profile_summary": data.get("profile_summary", "not mentioned")
    }

def get_company_role_context(state: CareerState) -> CareerState:
    query = f"""
Company: {state["company"]}
Role: {state["desired_role"]}
Eligibility criteria, required CGPA, required skills, job description,
study materials, interview questions
"""
    context = retrieve("placement", query)
    return {**state, "rag_context": context}

def check_eligibility(state: CareerState) -> CareerState:
    prompt = f"""
You are a Career Eligibility Checker.

Use only the retrieved context to check eligibility.

Retrieved Context:
{state["rag_context"]}

Student Profile:
Company: {state["company"]}
Desired Role: {state["desired_role"]}
CGPA: {state["cgpa"]}
Skills: {state["skills"]}
Experience: {state["experience"]}

Task:
Compare the student profile with the company and role requirements.

Return only valid JSON in this format:
{{
  "eligibility_status": "eligible or not_eligible or unclear",
  "missing_requirements": []
}}

Rules:
- If CGPA or required skills are missing in context, use "unclear".
- Do not assume company criteria if not present in context.
- missing_requirements must be a list.
"""
    response = llm.invoke(prompt)
    data = extract_json(get_text(response))
    return {
        **state,
        "eligibility_status": data.get("eligibility_status", "unclear"),
        "missing_requirements": data.get("missing_requirements", [])
    }

def generate_eligible_response(state: CareerState) -> CareerState:
    prompt = f"""
You are a Career Preparation Agent.

The student is eligible or mostly eligible.

Retrieved Context:
{state["rag_context"]}

Student Profile:
Company: {state["company"]}
Desired Role: {state["desired_role"]}
CGPA: {state["cgpa"]}
Skills: {state["skills"]}
Experience: {state["experience"]}

Generate a helpful answer with:
1. Eligibility result
2. Why the student is eligible
3. Study materials
4. Technical interview questions
5. HR interview questions
6. Preparation roadmap
7. Final advice

Keep the language simple and student friendly.
"""
    response = llm.invoke(prompt)
    return {**state, "final_answer": get_text(response)}

def generate_not_eligible_response(state: CareerState) -> CareerState:
    prompt = f"""
You are a Career Improvement Agent.

The student is not eligible or eligibility is unclear.

Retrieved Context:
{state["rag_context"]}

Student Profile:
Company: {state["company"]}
Desired Role: {state["desired_role"]}
CGPA: {state["cgpa"]}
Skills: {state["skills"]}
Experience: {state["experience"]}

Missing Requirements:
{state["missing_requirements"]}

Generate a helpful answer with:
1. Eligibility result
2. Missing requirements
3. Skill gap analysis
4. What to improve
5. Study materials
6. Interview questions for practice
7. 7 day improvement plan
8. Final advice

Keep the answer motivating and practical.
"""
    response = llm.invoke(prompt)
    return {**state, "final_answer": get_text(response)}

def decide_route(state: CareerState) -> str:
    if state["eligibility_status"].lower() == "eligible":
        return "eligible"
    return "not_eligible"

def build_career_agent():
    graph = StateGraph(CareerState)

    graph.add_node("infer_user_profile", infer_user_profile)
    graph.add_node("get_company_role_context", get_company_role_context)
    graph.add_node("check_eligibility", check_eligibility)
    graph.add_node("generate_eligible_response", generate_eligible_response)
    graph.add_node("generate_not_eligible_response", generate_not_eligible_response)

    graph.add_edge(START, "infer_user_profile")
    graph.add_edge("infer_user_profile", "get_company_role_context")
    graph.add_edge("get_company_role_context", "check_eligibility")

    graph.add_conditional_edges(
        "check_eligibility",
        decide_route,
        {
            "eligible": "generate_eligible_response",
            "not_eligible": "generate_not_eligible_response"
        }
    )

    graph.add_edge("generate_eligible_response", END)
    graph.add_edge("generate_not_eligible_response", END)

    return graph.compile()

if __name__ == "__main__":
    career_agent = build_career_agent()

    user_query = """
    I want to apply for TCS Ninja role.
    My CGPA is 7.8.
    I know Python, SQL, Machine Learning, FastAPI and basic DSA.
    Tell me whether I am eligible and how should I prepare.
    """

    result = career_agent.invoke({
        "user_input": user_query,
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

    print(result["final_answer"])