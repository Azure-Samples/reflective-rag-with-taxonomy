# Reflective RAG with Taxonomy

Many teams have hit the limits of Naive or Basic RAG. There are a number of approaches that improve RAG quality. This repo demonstrates some of them with a multi-agent approach.

## Features

The project includes demonstrations of:

* Taxonomy-based multi-agent research
* Reflection and evaluation of search results
* Automated taxonomy extraction from user queries
* Intelligent search query formulation
* Distributed research using LangGraph workflows
* Result consolidation with comprehensive answer generation

## Getting Started

### Prerequisites

- Python 3.12
- Azure Search service
- Azure OpenAI service
- Required environment variables:
  - AZURE_SEARCH_ENDPOINT
  - AZURE_SEARCH_KEY
  - AZURE_SEARCH_INDEX
  - AOAI_DEPLOYMENT
  - AOAI_KEY
  - AOAI_ENDPOINT

### Installation

```bash
# Clone the repository
git clone https://github.com/Azure-Samples/reflective-rag-with-taxonomy.git
cd reflective-rag-with-taxonomy

# Install dependencies
pip install -r requirements.txt
```

### Quickstart

1. Set up your .env file with the required Azure credentials
2. Ensure your Azure Search index is populated with documents
3. Run the multi-agent RAG system:

```bash
python backend/multi-agent-rag.py
```

## How It Works

The system uses a multi-agent approach to answer complex questions:

1. **Taxonomy Extraction**: Breaks down the user's question into 2-5 distinct taxonomies (categories) for research
2. **Distributed Research**: Assigns each taxonomy to a dedicated research agent
3. **Intelligent Search**: Each agent formulates optimized search queries for its taxonomy
4. **Result Evaluation**: Agents review search results to determine relevance and reliability
5. **Consolidation**: Results from all agents are combined
6. **Answer Generation**: A comprehensive response is created by synthesizing all research findings


## Architecture

The project uses:
- LangGraph for the agent workflow orchestration
- Azure Search for document retrieval (with both semantic and vector search)
- Azure OpenAI for language understanding and generation
- Pydantic for structured data validation

