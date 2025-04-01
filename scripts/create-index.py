# Suppress Azure SDK logging
import logging
logging.getLogger('azure').setLevel(logging.ERROR)
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.ERROR)

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchIndex
)
from datetime import datetime, timezone
import json
import hashlib
from typing import Any
#import lib for pypdf2


from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient  
from datetime import datetime
import os  
from dotenv import load_dotenv  
from azure.core.credentials import AzureKeyCredential  

from openai import AzureOpenAI  
import os
from langchain_openai import AzureChatOpenAI
import itertools



from azure.core.credentials import AzureKeyCredential

import tiktoken
from dotenv import load_dotenv 
import requests

load_dotenv()


ai_search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
ai_search_key = os.environ["AZURE_SEARCH_KEY"]
ai_search_index = os.environ["AZURE_SEARCH_INDEX"]

# Azure OpenAI
aoai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
aoai_key = os.getenv("AZURE_OPENAI_API_KEY")
aoai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")


search_index_client = SearchIndexClient(ai_search_endpoint, AzureKeyCredential(ai_search_key))
search_client = SearchClient(ai_search_endpoint, ai_search_index, AzureKeyCredential(ai_search_key))





def create_index():
    # Check if index exists, return if so
    try:
        search_index_client.get_index(ai_search_index)
        print("Index already exists")
        return
    except:
        pass

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, filterable=True,key=True),
        SimpleField(name="source_file", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source_pages", type=SearchFieldDataType.Collection(SearchFieldDataType.Int32)),
        SimpleField(name="sensitivity_label", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="created_date", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
        SimpleField(name="expiration_date", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
        SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=3072,
            vector_search_profile_name="myHnswProfile"
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw"
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            )
        ]
    )

    index = SearchIndex(name=ai_search_index, fields=fields, vector_search=vector_search)
    result = search_index_client.create_or_update_index(index)

    print("Index has been created")

if __name__ == "__main__":
    create_index()
