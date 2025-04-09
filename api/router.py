from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import importlib
from pydantic import BaseModel
from api.model import SearchRequest, SearchResultModel, TaxonomyRequest, MainStateModel
from api.service import  identify_taxonomies_api,process_research_workflow
multi_agent_rag = importlib.import_module("backend.multi-agent-rag")
internal_runagent_conversation = multi_agent_rag.run_multi_agent_conversation

router = APIRouter()

class QueryRequest(BaseModel):
    question: str

@router.post("/ask")
def ask_question(request: QueryRequest):
    try:
        result = process_research_workflow(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
#UI for the conversation and add LLm for intial start then call process_research_workflow
@router.post("/conversation")
async def start_research(user_input: str):
    try:
        # Start the research workflow
        state = process_research_workflow(user_input)

        # Return the final answer or consolidated results
        return {
            "final_answer": state["final_answer"],
            "thought_process": state["thought_process"],
            "research_results": state["research_results"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/taxonomies", response_model=MainStateModel)
def taxonomies_route(request: TaxonomyRequest):
    return identify_taxonomies_api(request)
