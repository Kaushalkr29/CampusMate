import os
from typing import TypedDict
 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END, StateGraph
 
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
 
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=api_key,
    temperature=0
)
 
# ---------------------------------------------------
# Startup Categories
# ---------------------------------------------------
 
STARTUP_QUERY_DATA = {
 
    "startup_query": {
        "keywords": [
            "business idea",
            "idea validation",
            "mca",
            "msme",
            "business model",
            "entrepreneurship"
        ],
        "priority": "medium"
    },
 
    "startup_knowledge": {
        "keywords": ["funding",
            "scheme",
            "government scheme",
            "incubator",
            "incubation",
            "innovation"
        ],
        "priority": "high"
    },
 
    "startup_recommendation": {
        "keywords": [
            "business plan",
            "roadmap",
            "market research",
            "recommendation"
        ],
        "priority": "medium"
    }
}
 
 
# ---------------------------------------------------
# State
# ---------------------------------------------------
 
class StartupState(TypedDict):
    query: str
    category: str
    priority: str
    recommendation: str
 
 
# ---------------------------------------------------
# Node 1
# Process startup-related queries
# ---------------------------------------------------
 
def process_startup_queries(state: StartupState) -> dict:
 
    query = state["query"].lower()
 
    for category, data in STARTUP_QUERY_DATA.items():
 
        for keyword in data["keywords"]:
 
            if keyword in query:
 
                return {
                    "category": category
                }
 
    return {
        "category": "startup_query"
    }
 
 
# ---------------------------------------------------
# Node 2
# Retrieve startup knowledge
# ---------------------------------------------------
 
def retrieve_startup_knowledge(state: StartupState) -> dict:
 
    category = state["category"]
 
    priority = STARTUP_QUERY_DATA[category]["priority"]
 
    return {
        "priority": priority
    }
 
 
# ---------------------------------------------------
# Node 3
# Generate startup recommendations
# ---------------------------------------------------
 
def generate_startup_recommendations(state: StartupState) -> dict:
 
    query = state["query"]
    category = state["category"]
    priority = state["priority"]
 
    prompt = f"""
You are a Startup Assistant.
 
Process startup-related queries:
Business Idea validation,
Startup documents (MCA, MSME),
Business model guidance,
Entrepreneurship support.
 
Retrieve startup knowledge:
Funding opportunities,
Government startup schemes,
Incubation programs,
Innovation resources.
 
Generate startup recommendations:
Business plan suggestions,
Funding roadmap,
Incubator recommendations,
Market research guidance.
 
User Query:
{query}
 
Category:
{category}
 
Priority:
{priority}
 
Give a clear and simple answer.
Keep the response within 150 words.
"""
 
    response = llm.invoke(prompt)
 
    return {
        "recommendation": response.content
    }
 
 
# ---------------------------------------------------
# Build Graph
# ---------------------------------------------------
 
def build_graph():
 
    builder = StateGraph(StartupState)
 
    builder.add_node(
        "process_startup_queries",
        process_startup_queries
    )
 
    builder.add_node(
        "retrieve_startup_knowledge",
        retrieve_startup_knowledge
    )
 
    builder.add_node(
        "generate_startup_recommendations",
        generate_startup_recommendations
    )
 
    builder.add_edge(
        START,
        "process_startup_queries"
    )
 
    builder.add_edge(
        "process_startup_queries",
        "retrieve_startup_knowledge"
    )
 
    builder.add_edge(
        "retrieve_startup_knowledge",
        "generate_startup_recommendations"
    )
 
    builder.add_edge(
        "generate_startup_recommendations",
        END
    )
 
    return builder.compile()
 
 
# ---------------------------------------------------
# Main Function
# ---------------------------------------------------
 
def main():
 
    user_query = input("Enter your query: ")
 
    state = {
        "query": user_query,
        "category": "",
        "priority": "",
        "recommendation": ""
    }
 
    graph = build_graph()
 
    result = graph.invoke(state)
 
    print("\nCategory :", result["category"])
    print("Priority :", result["priority"])
    print("\nRecommendation :", result["recommendation"])
 
if __name__ == "__main__":
    main()