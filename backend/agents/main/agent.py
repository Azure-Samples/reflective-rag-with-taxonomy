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


# Define the main graph invocation method
def graph_invoke(initial_state: ChatState) -> Generator[Dict[str, Any], None, ChatState]:
    """
    Master generator that chains the graph nodes created by build_main_graph.
    It loops through each step in the graph, updating the state as it proceeds.
    """
    # Build the main graph (workflow) to define the nodes and edges
    builder = build_main_graph()

    # Initialize the state with the initial input from the user
    state = initial_state
    current_node = "identify_taxonomies"  # Start with the first node

    while True:
        # Log current node and state
        print(f"Processing node: {current_node}")
        print(f"Current state: {state}")
        
        # Execute the current node's method in the workflow
        current_node_method = builder.get_node_method(current_node)
        result = current_node_method(state)
        
        # Update the state with the result of the node execution
        state = result

        # Log result
        print(f"Step result from {current_node}: {result}")

        # Determine the next node based on the edges defined in the graph
        next_nodes = builder.get_next_nodes(current_node)
        if not next_nodes:
            break  # If no next nodes, end the loop

        # Update the current node for the next iteration
        current_node = next_nodes[0]  # For simplicity, just take the first next node

        # Decision-making process
        decision = state["decisions"][-1] if state["decisions"] else "finalize"
        if decision == "finalize":
            break

    print(f"Final answer: {state['final_answer']}")
    return state

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


