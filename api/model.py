from typing import List, Optional, Set,Dict, Any

from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str
    processed_ids: Optional[Set[str]] = set()
    category_filter: Optional[str] = None

class SearchResultModel(BaseModel):
    id: str
    content: str
    source_file: str
    score: float

class TaxonomyRequest(BaseModel):
    user_input: str

class MainStateModel(BaseModel):
    user_input: str
    taxonomies: Optional[List[str]] = []
    research_outputs: Optional[List[dict]] = []
    thought_process: Optional[List[dict]] = []

class MainState(BaseModel):
    user_input: str
    taxonomies: Optional[List[str]] = []
    research_results: List[dict] = []
    final_answer: Optional[str] = None
    thought_process: Optional[List[dict]] = []
    research_outputs: Optional[List[dict]] = []

class ResearchState:
    def __init__(self, taxonomy: str):
        self.taxonomy = taxonomy
        self.search_query = ""
        self.current_results: List[Dict[str, Any]] = []
        self.vetted_results: List[Dict[str, Any]] = []
