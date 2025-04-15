from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from langsmith import traceable
from langgraph.graph import StateGraph, START, END

from backend.utils.llm import LLM
from backend.utils.classes import *
import backend.agents.research.prompts as prompts

from typing import List, Set
from dotenv import load_dotenv
import os
import time


load_dotenv(dotenv_path="example.env")

AI_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
AI_SEARCH_INDEX = os.environ["AZURE_SEARCH_INDEX"]
AI_SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]

K_NEAREST_NEIGHBORS = int(os.environ["K_NEAREST_NEIGHBORS"])
NUM_SEARCH_RESULTS = int(os.environ["NUM_SEARCH_RESULTS"])
MAX_ATTEMPTS = int(os.environ["MAX_ATTEMPTS"])

class ReviewLLM(LLM):
    _SEARCH_RESULT = SearchClient(AI_SEARCH_ENDPOINT, AI_SEARCH_INDEX, AzureKeyCredential(AI_SEARCH_KEY))
    
    def __init__(self):
        super().__init__()
        self.__model = LLM._llm_model.with_structured_output(ReviewDecision)
        self.__search_client = SearchClient(AI_SEARCH_ENDPOINT, AI_SEARCH_INDEX, AzureKeyCredential(AI_SEARCH_KEY))
        self.__research_graph = self.__build_research_graph()
    
    def __format_search_results(self,results: List[_SEARCH_RESULT]) -> str:
        """Format search results into a nicely formatted string."""
        output_parts = ["\n=== Search Results ==="]
        for i, result in enumerate(results, 0):
            result_parts = [
                f"\nResult #{i}",
                "=" * 80,
                f"ID: {result['id']}",
                f"Source File: {result['source_file']}",
                #f"Source Pages: {result['source_pages']}",
                "\n<Start Content>",
                "-" * 80,
                result['content'],
                "-" * 80,
                "<End Content>"
            ]
            output_parts.extend(result_parts)
        formatted_output = "\n".join(output_parts)
        return formatted_output
    
    @traceable(run_type="retriever", name="run_search")
    def __run_search(self,search_query: str, processed_ids: Set[str], category_filter: str | None = None) -> List[_SEARCH_RESULT]:
        """
        Perform a search using Azure Cognitive Search with both semantic and vector queries.
        """
        # Generate vector embedding for the query
        query_vector = self._embeddings_model.embed_query(search_query)
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=K_NEAREST_NEIGHBORS,
            fields="content_vector"
        )
        
        # Create filter combining processed_ids and category filter
        filter_parts = []
        if processed_ids:
            ids_string = ','.join(processed_ids)
            filter_parts.append(f"not search.in(id, '{ids_string}')")
        if category_filter:
            filter_parts.append(f"({category_filter})")
        filter_str = " and ".join(filter_parts) if filter_parts else None

        # Perform the search
        results = self.__search_client.search(
            search_text=search_query,
            vector_queries=[vector_query],
            filter=filter_str,
            select=["id", "content", "source_file"], #, "source_pages"
            top=NUM_SEARCH_RESULTS
        )
        
        search_results = []
        for result in results:
            search_result = SearchResult(
                id=result["id"],
                content=result["content"],
                source_file=result["source_file"],
                #source_pages=result["source_pages"],
                score=result["@search.score"]
            )
            search_results.append(search_result)
        
        return search_results

    async def __review_results(self, state: ResearchState) -> ResearchState | ResearchOutputState:
        """Review current results and categorize them as valid or invalid.
        When review decision is 'finalize', return the final output directly."""
        
        await self.__push_updates(message_source="Research Agent", push_update= f"Evaluating search attempt {state['attempts']} for taxonomy: {state['taxonomy']}")

        review_prompt = prompts.REVIEW_PROMPT

        current_results_formatted = self.__format_search_results(state["current_results"]) if state["current_results"] else "No current results."
        vetted_results_formatted = self.__format_search_results(state["vetted_results"]) if state["vetted_results"] else "No previously vetted results."
        
        search_history_formatted = ""
        if state["search_history"]:
            search_history_formatted = "\n### Search History ###\n"
            for i, (search, review) in enumerate(zip(state["search_history"], state["reviews"]), 1):
                search_history_formatted += f"<Attempt {i}>\n"
                search_history_formatted += f"   Query: {search['query']}\n"
                search_history_formatted += f"   Filter: {search['filter']}\n"
                search_history_formatted += f"   Review: {review}\n"
        
        llm_input = review_prompt.format(
            question=state["user_input"],
            history=state["user_history"],
            taxonomy=state["taxonomy"],
            current_results=current_results_formatted,
            vetted_results=vetted_results_formatted,
            search_history=search_history_formatted
        )
        
        messages = [
            {"role": "system", "content": "You are an expert at evaluating search results."},
            {"role": "user", "content": llm_input}
        ]
        
        review = self.__model.invoke(messages)
        
        # Add to thought process
        state["thought_process"].append({
            "type": "review",
            "details": {
                "taxonomy": state["taxonomy"],
                "thought_process": review.thought_process,
                "decision": review.decision,
                "valid_results": len(review.valid_results),
                "invalid_results": len(review.invalid_results)
            }
        })
        
        state["reviews"].append(review.thought_process)
        state["decisions"].append(review.decision)
        
        for idx in review.valid_results:
            result = state["current_results"][idx]
            state["vetted_results"].append(result)
            state["processed_ids"].add(result["id"])
        
        for idx in review.invalid_results:
            result = state["current_results"][idx]
            state["discarded_results"].append(result)
            state["processed_ids"].add(result["id"])
        
        state["current_results"] = []
        
        # If maximum attempts reached or decision is finalize, return the final output
        if review.decision == "finalize" or state["attempts"] >= MAX_ATTEMPTS:
            await self.__push_updates(message_source="Research Agent", push_update= f"Finalizing research for taxonomy: {state['taxonomy']}")
            
            # Create a result dictionary for this taxonomy
            taxonomy_result = {
                "taxonomy": state["taxonomy"],
                "vetted_results": state["vetted_results"]
            }
            
            # Return a ResearchOutputState with research_outputs that can be merged with the main state
            return ResearchOutputState(
                research_outputs=[taxonomy_result]  # List with a single item that will be merged with the main state
            )
        
        # Otherwise, continue with the research loop
        return state

    # Build research agent graph
    def __build_research_graph(self):
        """Build the research agent workflow graph"""
        builder = StateGraph(ResearchState, output=ResearchOutputState)
        
        # Add nodes
        builder.add_node("generate_search_query", self.__generate_search_query)
        builder.add_node("review_results", self.__review_results)
        
        # Add edges
        builder.add_edge(START, "generate_search_query")
        builder.add_edge("generate_search_query", "review_results")
        
        # Add conditional edges based on review outcome
        builder.add_conditional_edges(
            "review_results",
            self.__review_router,
            {
                "retry": "generate_search_query",
                "finalize": END
            }
        )
        
        return builder.compile()
    
    def get_research_graph(self):
        return self.__research_graph
    
    async def __generate_search_query(self, state: ResearchState) -> ResearchState:
        """Generate an optimized search query based on the current state"""
        await self.__push_updates(message_source="Research Agent", push_update= f"Generating search query for taxonomy: {state['taxonomy']}")
        state["attempts"] += 1
        
        query_prompt = prompts.QUERY_PROMPT
        
        search_history_formatted = ""
        if state["search_history"]:
            search_history_formatted = "\n### Previous Search Attempts ###\n"
            for i, (search, review) in enumerate(zip(state["search_history"], state["reviews"]), 1):
                search_history_formatted += f"<Attempt {i}>\n"
                search_history_formatted += f"   Query: {search['query']}\n"
                search_history_formatted += f"   Filter: {search['filter']}\n"
                search_history_formatted += f"   Review: {review}\n"
        
        llm_input = query_prompt.format(
            question=state["user_input"],
            taxonomy=state["taxonomy"],
            search_history=search_history_formatted
        )
        
        messages = [
            {"role": "system", "content": "You are an expert search query generator."},
            {"role": "user", "content": llm_input}
        ]
        
        llm_with_search_prompt = self._llm_model.with_structured_output(SearchPromptResponse)
        search_response = llm_with_search_prompt.invoke(messages)
        
        # Record this search query in history
        state["search_history"].append({
            "query": search_response.search_query,
            "filter": search_response.filter
        })
        
        # Run the search
        current_results = self.__run_search(
            search_query=search_response.search_query,
            processed_ids=state["processed_ids"],
            category_filter=search_response.filter
        )
        state["current_results"] = current_results
        
        # Add to thought process
        state["thought_process"].append({
            "type": "search_query",
            "details": {
                "taxonomy": state["taxonomy"],
                "query": search_response.search_query,
                "filter": search_response.filter,
                "num_results": len(current_results)
            }
        })
        
        return state
    
    async def __review_router(self, state: ResearchState) -> str:
        """Route to either retry search or go to END (finalize happens in review_results now)"""
        if state["attempts"] >= MAX_ATTEMPTS:
            await self.__push_updates(message_source="Research Agent", push_update= f"\nReached maximum attempts ({MAX_ATTEMPTS}) for taxonomy {state['taxonomy']}. Proceeding to finalize.")
            return "finalize"
        
        latest_decision = state["decisions"][-1]
        if latest_decision == "finalize":
            return "finalize"
        
        return "retry"
    
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