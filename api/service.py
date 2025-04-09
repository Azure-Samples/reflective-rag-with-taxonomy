from typing import List, Set,Dict, Any
import importlib
from pydantic import BaseModel
from fastapi import HTTPException

multi_agent_rag = importlib.import_module("backend.multi-agent-rag")

internal_identify_taxonomies = multi_agent_rag.identify_taxonomies
internal_runagent_conversation = multi_agent_rag.run_multi_agent_conversation

from api.model import SearchRequest, SearchResultModel, TaxonomyRequest, MainStateModel,MainState, ResearchState

class QueryRequest(BaseModel):
    question: str

def process_research_workflow(user_input: str) -> MainState:
   result = internal_runagent_conversation(user_input)
   
   return format_final_state_for_ui(result)


def identify_taxonomies_api(request: TaxonomyRequest) -> MainStateModel:
    state = {
        "user_input": request.user_input,
        "thought_process": []
    }
    updated_state = internal_identify_taxonomies(state)
    return MainStateModel(**updated_state)

def format_final_state_for_ui(final_state: Dict[str, Any]) -> Dict[str, Any]:
    def format_research_results(results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        return [
            {
                "snippet": res["content"].strip(),
                "source": res["source_file"],
                "confidence": f"{res['score']:.2%}"
            }
            for res in results
        ]

    formatted_output = {
        "userQuery": final_state.get("user_input", ""),
        "finalAnswer": final_state.get("final_answer", ""),
        "taxonomySummary": [],
        "thoughtProcess": [],
    }

    # Format taxonomies and research results
    for entry in final_state.get("research_results", []):
        taxonomy_name = entry.get("taxonomy")
        vetted = format_research_results(entry.get("vetted_results", []))
        formatted_output["taxonomySummary"].append({
            "taxonomy": taxonomy_name,
            "results": vetted if vetted else ["No relevant information found."]
        })

    # Thought process
    for thought in final_state.get("thought_process", []):
        t_type = thought.get("type", "")
        details = thought.get("details", {})
        if t_type == "taxonomy_extraction":
            formatted_output["thoughtProcess"].append(
                f"Identified taxonomies: {', '.join(details.get('taxonomies', []))}."
            )
        elif t_type == "consolidation":
            formatted_output["thoughtProcess"].append(
                f"Consolidated results across {details.get('num_taxonomies', 0)} taxonomies."
            )
        elif t_type == "final_answer":
            formatted_output["thoughtProcess"].append(
                f"Synthesized final answer: {details.get('final_answer', '')}"
            )
        else:
            formatted_output["thoughtProcess"].append(f"{t_type.capitalize()} step processed.")

    return formatted_output

