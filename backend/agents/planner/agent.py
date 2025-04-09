from backend.utils.llm import LLM
from backend.utils.classes import *
import backend.agents.planner.prompts as prompts

from langgraph.constants import Send

class TaxonomyLLM(LLM):
    def __init__(self):
        super().__init__()
        self.__model = LLM._llm_model.with_structured_output(TaxonomyExtraction)
    
    def identify_taxonomies(self,state: MainState) -> MainState:
        """Extract taxonomies from the user's question"""
        print('Identifying taxonomies from the question')
        
        taxonomy_prompt = prompts.TAXONOMY_PROMPT
        
        messages = [
            {"role": "system", "content": taxonomy_prompt},
            {"role": "user", "content": f"Extract taxonomies from this question: {state['user_input']}"}
        ]
        
        taxonomy_response = self.__model.invoke(messages)
        state["taxonomies"] = taxonomy_response.taxonomies
        
        # Add to thought process
        state["thought_process"].append({
            "type": "taxonomy_extraction",
            "details": {
                "taxonomies": taxonomy_response.taxonomies,
                "reasoning": taxonomy_response.reasoning
            }
        })
        
        print(f"Identified taxonomies: {state['taxonomies']}")
        
        return state
    
    def distribute_research_tasks(self,state: MainState) -> list:
        """Distribute research tasks to individual research agents based on taxonomies"""
        print(f"Distributing {len(state['taxonomies'])} research tasks to agents")
        
        return [
            Send("research_agent", {
                "taxonomy": taxonomy,
                "user_input": state["user_input"],
                "current_results": [],
                "vetted_results": [],
                "discarded_results": [],
                "processed_ids": set(),
                "reviews": [],
                "decisions": [],
                "attempts": 0,
                "search_history": [],
                "thought_process": []
            }) for taxonomy in state["taxonomies"]
        ]