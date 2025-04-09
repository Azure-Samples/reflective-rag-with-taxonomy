from backend.utils.classes import *
from backend.utils.llm import LLM
import backend.agents.consolidation.prompts as prompts

class Consolidate():

    def consolidate_results(self,state: MainState) -> MainState:
        """Consolidate results from all research agents"""
        print("Consolidating results from all research agents")
        
        # The research_outputs are automatically collected via the Annotated field
        # These will be a list of dictionaries with taxonomy and vetted_results
        research_outputs = state["research_outputs"]
        
        # Add to thought process
        state["thought_process"].append({
            "type": "consolidation",
            "details": {
                "num_taxonomies": len(state["taxonomies"]),
                "results_per_taxonomy": {result["taxonomy"]: len(result["vetted_results"]) for result in research_outputs}
            }
        })
        
        # Store research results for final inference
        state["research_results"] = research_outputs
        
        return state

    def final_inference(self,state: MainState) -> MainState:
        """Generate final answer by synthesizing all research results"""
        print("Generating final answer")
        
        final_prompt = prompts.CONSOLIDATION_PROMPT
        
        # Format research results for the prompt
        formatted_results = ""
        for result in state["research_results"]:
            formatted_results += f"\n=== Taxonomy: {result['taxonomy']} ===\n"
            if result["vetted_results"]:
                for i, res in enumerate(result["vetted_results"]):
                    formatted_results += f"Result {i+1}: {res['content']}\n"
            else:
                formatted_results += "No relevant results found for this taxonomy.\n"
        
        messages = [
            {"role": "system", "content": final_prompt},
            {"role": "user", "content": final_prompt.format(
                question=state["user_input"],
                research_results=formatted_results
            )}
        ]
        
        response_chunks = []
        for chunk in LLM._llm_model.stream(messages): #might not be best practice
            response_chunks.append(chunk.content)
            print(chunk.content, end="", flush=True)
        
        final_answer = "".join(response_chunks)
        state["final_answer"] = final_answer
        
        # Add to thought process
        state["thought_process"].append({
            "type": "final_answer",
            "details": {
                "final_answer": final_answer
            }
        })
        
        return state