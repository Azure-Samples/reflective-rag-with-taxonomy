from pydantic import BaseModel
from typing import List, Dict, Any, Literal, Set, TypedDict, Annotated
import operator
import asyncio

data_queue = asyncio.Queue()

NUM_SEARCH_RESULTS = 5

# Create a type for indices from 0 to NUM_SEARCH_RESULTS-1
SearchResultIndex = Literal[tuple(range(NUM_SEARCH_RESULTS))]

# Type Definitions
class SearchResult(TypedDict):
    id: str
    content: str
    source_file: str
    source_pages: int
    score: float

class QuestionRequest(BaseModel):
    user_input: str
    history: str

class ReviewDecision(BaseModel):
    """Schema for review agent decisions"""

    thought_process: str
    valid_results: List[SearchResultIndex]  # Indices of valid results
    invalid_results: List[SearchResultIndex]  # Indices of invalid results
    decision: Literal["retry", "finalize"]


class SearchPromptResponse(BaseModel):
    """Schema for search prompt responses"""

    search_query: str
    filter: str | None


class TaxonomyExtraction(BaseModel):
    """Schema for taxonomy extraction"""

    taxonomies: List[str]
    reasoning: str


# Main state for the overall workflow
class MainState(TypedDict):
    user_input: str
    user_history: str
    taxonomies: List[str]
    research_results: List[Dict[str, Any]]
    final_answer: str | None
    thought_process: List[Dict[str, Any]]  # List of thought process steps
    # Following ReportState pattern from report builder
    research_outputs: Annotated[List[Dict[str, Any]], operator.add]


# Output state for research agents
class ResearchOutputState(TypedDict):
    research_outputs: List[Dict[str, Any]]  # This field needs to match one in MainState


# State for individual research agents
class ResearchState(TypedDict):
    taxonomy: str
    user_input: str
    user_history: str
    current_results: List[SearchResult]
    vetted_results: List[SearchResult]
    discarded_results: List[SearchResult]
    processed_ids: Set[str]  # Track all processed document IDs
    reviews: List[str]  # Thought processes from reviews
    decisions: List[str]  # Store the actual decisions
    attempts: int  # Track number of search attempts
    search_history: List[Dict[str, Any]]  # Track previous search queries and filters
    thought_process: List[Dict[str, Any]]  # List of thought process steps

class ChatState(TypedDict):
    user_input: str
    current_results: List[Any]
    vetted_results: List[Any]
    discarded_results: List[Any]
    processed_ids: Set[str]  # Track all processed document IDs
    reviews: List[str]       # Thought processes from reviews
    decisions: List[str]     # Store the actual decisions
    final_answer: str | None
    attempts: int            # Track number of search attempts
    search_history: List[Dict[str, Any]]  # Track previous search queries and filters
    thought_process: List[Dict[str, Any]]  # List of thought process steps
