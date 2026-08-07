import os
from typing import TypedDict

from dotenv import load_dotenv
from ddgs import DDGS

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END, StateGraph

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
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
        "keywords": [
            "funding",
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
    user_query: str
    category: str
    priority: str
    search_results: str
    feasibility: str
    recommendation: str

# ---------------------------------------------------
# Node 1
# Search Idea Online
# ---------------------------------------------------

def search_business_idea(state: StartupState):

    query = state["user_query"]

    results = []

    try:

        with DDGS() as ddgs:

            for r in ddgs.text(query, max_results=5):

                results.append(
                    f"""
Title: {r.get('title', '')}

Snippet: {r.get('body', '')}

Link: {r.get('href', '')}
"""
                )

    except Exception as e:

        results.append(f"Search Error: {str(e)}")

    return {
        "search_results": "\n".join(results)
    }

# ---------------------------------------------------
# Node 2
# Validate Feasibility
# ---------------------------------------------------

def validate_business_idea(state: StartupState):

    prompt = f"""
You are a Startup Analyst.

Business Idea:
{state["user_query"]}

Online Search Results:
{state["search_results"]}

Analyze:

1. Market Demand
2. Competition
3. Practicality
4. Scalability
5. Business Potential

Return ONLY ONE WORD:

feasible

or

not_feasible
"""

    response = llm.invoke(prompt)

    return {
        "feasibility": response.content[0]["text"]
    }

# ---------------------------------------------------
# Router
# ---------------------------------------------------

def feasibility_router(state: StartupState):

    if "feasible" == state["feasibility"].strip().lower():
        return "continue"

    return "stop"

# ---------------------------------------------------
# Node 3
# Process Startup Query
# ---------------------------------------------------

def process_startup_queries(state: StartupState):

    user_query = state["user_query"].lower()

    for category, data in STARTUP_QUERY_DATA.items():

        for keyword in data["keywords"]:

            if keyword in user_query:

                return {
                    "category": category
                }

    return {
        "category": "startup_query"
    }

# ---------------------------------------------------
# Node 4
# Retrieve Startup Knowledge
# ---------------------------------------------------

def retrieve_startup_knowledge(state: StartupState):

    category = state["category"]

    priority = STARTUP_QUERY_DATA[category]["priority"]

    return {
        "priority": priority
    }

# ---------------------------------------------------
# Node 5
# Generate Recommendation
# ---------------------------------------------------

def generate_startup_recommendations(state: StartupState):

    prompt = f"""
You are a Startup Assistant.

Business Idea:
{state["user_query"]}

Category:
{state["category"]}

Priority:
{state["priority"]}

Online Research:
{state["search_results"]}

Generate:

1. Feasibility Summary

2. Recommended Business Model

3. Target Customers

4. Revenue Sources

5. Market Research Suggestions

6. Funding Opportunities

7. Government Schemes (if relevant)

8. Incubator Recommendations

9. First 90-Day Startup Roadmap

Keep response practical and simple.
"""

    response = llm.invoke(prompt)

    return {
        "recommendation": response.content[0]["text"]
    }

# ---------------------------------------------------
# Node 6
# Reject Idea
# ---------------------------------------------------

def reject_business_idea(state: StartupState):

    return {
        "recommendation":
        """
The business idea appears to be NOT FEASIBLE based on available online information.

Possible reasons:
- Insufficient market demand
- Technical limitations
- Economic challenges
- Legal or operational constraints

Please refine the business idea and try again.
"""
    }

# ---------------------------------------------------
# Build Graph
# ---------------------------------------------------

def build_graph():

    builder = StateGraph(StartupState)

    builder.add_node(
        "search_business_idea",
        search_business_idea
    )

    builder.add_node(
        "validate_business_idea",
        validate_business_idea
    )

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

    builder.add_node(
        "reject_business_idea",
        reject_business_idea
    )

    builder.add_edge(
        START,
        "search_business_idea"
    )

    builder.add_edge(
        "search_business_idea",
        "validate_business_idea"
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

    builder.add_edge(
        "reject_business_idea",
        END
    )

    builder.add_conditional_edges(
        "validate_business_idea",
        feasibility_router,
        {
            "continue": "process_startup_queries",
            "stop": "reject_business_idea"}
    )
    return builder.compile()

