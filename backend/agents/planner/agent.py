from backend.utils.llm import LLM
from backend.utils.classes import *
import backend.agents.planner.prompts as prompts
import time

from langgraph.constants import Send

class TaxonomyLLM(LLM):
    def __init__(self):
        super().__init__()
        self.__model = LLM._llm_model.with_structured_output(TaxonomyExtraction)
    
    async def identify_taxonomies(self,state: MainState) -> MainState:
        """Extract taxonomies from the user's question"""
        
        await self.__push_updates(message_source="Planner Agent", push_update="Extracting taxonomies from the user question")
        
        taxonomy_prompt = prompts.TAXONOMY_PROMPT
        
        messages = [
            {"role": "system", "content": taxonomy_prompt},
           {"role": "user", "content": f"Extract taxonomies from this question: {state['user_input']}. Make sure to take into consideration this chat history:{state['user_history']}"}
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
        
        await self.__push_updates(message_source="Planner Agent", push_update=f"Identified taxonomies from the user question: {state['taxonomies']}")
        
        return state
    
    async def distribute_research_tasks(self,state: MainState) -> list:
        """Distribute research tasks to individual research agents based on taxonomies"""
        
        await self.__push_updates(message_source="Planner Agent", push_update="Initiating research agents for each extracted taxonomy")
        
        return [
            Send("research_agent", {
                "taxonomy": taxonomy,
                "user_input": state["user_input"],
                "user_history": state["user_history"],
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
    
    async def __push_updates(self, message_source: str, push_update: str) -> None:
        """Push updates to the user"""
        # Implement a mechanism to send update messages to the user
        current_time = time.time()

        await data_queue.put({
            "message_source": message_source,
            "message_content": push_update,
            "message_timestamp": current_time,
        })

        print(f"UX UPDATE - Agent Type: {message_source}, Message: {push_update}, Message Time: {current_time}")