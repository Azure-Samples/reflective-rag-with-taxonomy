from typing import Generator, Dict, Any, List, Set, TypedDict

from backend.utils.classes import MainState,ChatState
from backend.agents.consolidation.agent import Consolidate
from backend.agents.research.agent import ReviewLLM
from backend.agents.planner.agent import TaxonomyLLM

from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="example.env")

aoai_deployment = os.getenv("OPENAI_API_VERSION")
aoai_key = os.getenv("AZURE_OPENAI_API_KEY")
aoai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")


def build_main_graph():
    """Build the main workflow graph"""

    consolidate_agent = Consolidate()
    review_agent = ReviewLLM()
    taxonomy_agent = TaxonomyLLM()

    # Initialize graph with MainState which has the Annotated field for research_outputs
    builder = StateGraph(MainState)
    builder.add_node("identify_taxonomies", taxonomy_agent.identify_taxonomies)
    builder.add_node("research_agent", review_agent.get_research_graph())
    builder.add_node("consolidate_results", consolidate_agent.consolidate_results)
    builder.add_node("final_inference", consolidate_agent.final_inference)
    
    # Using the same pattern as the report builder
    builder.add_edge(START, "identify_taxonomies")
    builder.add_conditional_edges(
        "identify_taxonomies",
        taxonomy_agent.distribute_research_tasks,
        ["research_agent"]
    )
    
    # Note: No set_field here - we're using the Annotated field pattern from ReportState
    
    builder.add_edge("research_agent", "consolidate_results")
    builder.add_edge("consolidate_results", "final_inference")
    builder.add_edge("final_inference", END)
    
    return builder.compile()


