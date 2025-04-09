"""
This module combines document processing, chunking, and indexing functionality.
It processes documents from Azure Data Lake Storage or local filesystem using Document Intelligence,
chunks them while maintaining page context, and uploads to Azure Cognitive Search.
"""

import os
import hashlib
from typing import List, Dict, Any
from dotenv import load_dotenv
from document_processing import (
    get_document_intelligence_client, 
    get_blob_service_client, 
    analyze_document,
    analyze_local_document
)
from chunking import recursive_character_chunking_langchain, semantic_chunking_langchain, chunk_by_tokens_langchain
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import json
from datetime import datetime, timezone
from langchain_openai import AzureOpenAIEmbeddings

# Load environment variables
load_dotenv()

# Azure Configuration
STORAGE_ACCOUNT_NAME = os.environ.get("STORAGE_ACCOUNT_NAME")
STORAGE_ACCOUNT_CONTAINER = os.environ.get("STORAGE_ACCOUNT_CONTAINER")
AI_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
AI_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
AI_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX")

aoai_endpoint = os.environ.get("AOAI_ENDPOINT")
aoai_key = os.environ.get("AOAI_KEY")

embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-large",
    api_key=aoai_key,
    azure_endpoint=aoai_endpoint
)

def get_metadata(filename: str) -> Dict[str, str]:
    """
    Get metadata for a document based on filename.
    
    This function returns hardcoded values for now but could be expanded
    to retrieve metadata from a database, API, or more sophisticated logic.
    
    Parameters
    ----------
    filename : str
        Name of the file to get metadata for
        
    Returns
    -------
    Dict[str, str]
        Dictionary containing taxonomy and sensitivity_label
    """
    
    # Default values for unknown file types
    return {
        "taxonomy": "test",
        "sensitivity_label": "internal"
    }

class DocumentProcessor:
    def __init__(self):
        """
        Initialize the document processor with necessary clients.
        """
        self.doc_intelligence_client = get_document_intelligence_client()
        self.blob_service_client = get_blob_service_client()
        self.search_client = SearchClient(
            AI_SEARCH_ENDPOINT,
            AI_SEARCH_INDEX,
            AzureKeyCredential(AI_SEARCH_KEY)
        )
        
        print("\nDocument processor initialized")
        print("Using dynamic metadata assignment for each document")

    def process_document(self, blob_name: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        """
        Process a single document from ADLS:
        1. Analyze with Document Intelligence
        2. Chunk the content while maintaining page numbers
        3. Upload chunks to the search index
        
        Parameters
        ----------
        blob_name : str
            The name of the blob in ADLS to process
        chunk_size : int, optional
            The target size of each chunk in characters
        chunk_overlap : int, optional
            The number of characters to overlap between chunks
        """
        print(f"Processing document from ADLS: {blob_name}")
        
        # Get metadata for this file
        filename = os.path.basename(blob_name)
        metadata = get_metadata(filename)
        
        taxonomy = metadata["taxonomy"]
        sensitivity_label = metadata["sensitivity_label"]
            
        print(f"Using metadata for {filename}: taxonomy={taxonomy}, sensitivity={sensitivity_label}")
        
        # Generate blob URL
        blob_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{STORAGE_ACCOUNT_CONTAINER}/{blob_name}"
        
        # Analyze document with Document Intelligence
        print("Analyzing document with Document Intelligence")
        analyze_request = {"urlSource": blob_url}
        poller = self.doc_intelligence_client.begin_analyze_document("prebuilt-layout", analyze_request=analyze_request)
        result = poller.result()

        # Extract text with page numbers
        full_text = self._extract_text_with_page_numbers(result)

        # Process the text and upload chunks
        self._process_and_upload_chunks(
            full_text=full_text,
            source_id=blob_name,
            taxonomy=taxonomy,
            sensitivity_label=sensitivity_label
        )

    def process_local_document(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        """
        Process a single document from local filesystem:
        1. Analyze with Document Intelligence
        2. Chunk the content while maintaining page numbers
        3. Upload chunks to the search index
        
        Parameters
        ----------
        file_path : str
            The path to the local file to process
        chunk_size : int, optional
            The target size of each chunk in characters
        chunk_overlap : int, optional
            The number of characters to overlap between chunks
        """
        print(f"Processing local document: {file_path}")
        
        # Extract filename from path for metadata lookup
        filename = os.path.basename(file_path)
        
        # Get metadata for this file
        filename = os.path.basename(file_path)
        metadata = get_metadata(filename)
        
        taxonomy = metadata["taxonomy"]
        sensitivity_label = metadata["sensitivity_label"]
            
        print(f"Using metadata for {filename}: taxonomy={taxonomy}, sensitivity={sensitivity_label}")
        
        # Analyze document with Document Intelligence
        print("Analyzing document with Document Intelligence")
        result = analyze_local_document(file_path)

        # Extract text with page numbers
        full_text = self._extract_text_with_page_numbers(result)

        # Process the text and upload chunks
        self._process_and_upload_chunks(
            full_text=full_text,
            source_id=filename,
            taxonomy=taxonomy,
            sensitivity_label=sensitivity_label
        )

    def _extract_text_with_page_numbers(self, result) -> str:
        """
        Extract text from Document Intelligence result and add page markers.
        
        Parameters
        ----------
        result : AnalyzeResult
            The result from Document Intelligence
            
        Returns
        -------
        str
            The full text with page markers
        """
        full_text = ""
        page_number = 1
        for page in result.pages:
            page_text = ""
            for line in page.lines:
                page_text += line.content + "\n"
            # Add page marker at the end of each page
            page_text += f'###Page Number: {page_number}###\n\n'
            full_text += page_text
            page_number += 1
        return full_text

    def _process_and_upload_chunks(self, full_text: str, source_id: str, taxonomy: str, sensitivity_label: str) -> None:
        """
        Process text into chunks and upload to Azure Cognitive Search.
        
        Parameters
        ----------
        full_text : str
            The full text with page markers
        source_id : str
            The identifier for the source document (filename or blob name)
        taxonomy : str
            The document taxonomy classification
        sensitivity_label : str
            The document sensitivity label
        """
        # Chunk the document
        print("Chunking document")
        # chunks = recursive_character_chunking_langchain(full_text)
        # chunks = semantic_chunking_langchain(full_text)
        chunks = chunk_by_tokens_langchain(full_text)

        # Process and upload chunks
        documents = []
        current_page = 1
        
        for i, chunk in enumerate(chunks):
            # Find page numbers in chunk
            page_numbers = []
            lines = chunk.split('\n')
            for line in lines:
                if '###Page Number:' in line:
                    try:
                        page_num = int(line.split('###Page Number:')[1].split('###')[0].strip())
                        page_numbers.append(page_num)
                    except ValueError:
                        continue

            # Determine page range for chunk
            if page_numbers:
                chunk_start_page = page_numbers[0]
                chunk_end_page = page_numbers[-1] if len(page_numbers) > 1 else page_numbers[0]
                current_page = chunk_end_page
            else:
                chunk_start_page = current_page
                chunk_end_page = current_page

            # Generate unique ID for chunk
            chunk_id = hashlib.md5((source_id + str(i)).encode()).hexdigest()

            # Generate vector embedding for chunk
            try:
                content_vector = embeddings_model.embed_query(chunk)
            except Exception as e:
                print(f"Error generating vector embedding for chunk {chunk_id} in {source_id}: {str(e)}")
                continue

            # Create document for indexing with metadata
            document = {
                "id": chunk_id,
                "source_file": source_id,
                "source_pages": [p for p in range(chunk_start_page, chunk_end_page + 1)],
                "content": chunk,
                "content_vector": content_vector,
                "taxonomy": taxonomy,
                "sensitivity_label": sensitivity_label,
                "created_date": datetime.now(timezone.utc).isoformat()
            }
            documents.append(document)

        # Upload chunks to search index
        print(f"Uploading {len(documents)} chunks to search index")
        # Upload in batches of 100 to avoid service limits
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            self.search_client.upload_documents(batch)
        print(f"Successfully processed and indexed document: {source_id}")

    def process_all_documents(self) -> None:
        """Process all documents in the configured ADLS container."""
        container_client = self.blob_service_client.get_container_client(STORAGE_ACCOUNT_CONTAINER)
        
        for blob in container_client.list_blobs():
            try:
                self.process_document(blob.name)
            except Exception as e:
                print(f"Error processing document {blob.name}: {str(e)}")
                continue

    def process_all_local_documents(self, directory_path: str) -> None:
        """
        Process all documents in the specified local directory.
        
        Parameters
        ----------
        directory_path : str
            Path to the directory containing documents to process
        """
        if not os.path.isdir(directory_path):
            print(f"Error: {directory_path} is not a valid directory")
            return
            
        # Get list of supported file extensions
        supported_extensions = ['.pdf', '.docx', '.doc', '.pptx', '.xlsx', '.jpg', '.jpeg', '.png', '.tiff', '.tif']
        
        print(f"Processing all documents in directory: {directory_path}")
        
        # Count how many files we'll process
        total_files = sum(1 for root, _, files in os.walk(directory_path) 
                         for file in files 
                         if any(file.lower().endswith(ext) for ext in supported_extensions))
        print(f"Found {total_files} document(s) to process")
        
        processed_count = 0
        error_count = 0
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                # Check if file has supported extension
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        print(f"Processing {processed_count + 1}/{total_files}: {file}")
                        self.process_local_document(file_path)
                        processed_count += 1
                    except Exception as e:
                        print(f"Error processing document {file_path}: {str(e)}")
                        error_count += 1
                        continue
        
        print(f"Processing complete. Processed {processed_count} document(s) with {error_count} error(s).")

def main():
    """Main function to run the document processing pipeline."""
    processor = DocumentProcessor()
    
    # Choose one of these options:
    
    # Option 1: Process from Azure storage
    # processor.process_all_documents()
    
    # Option 2: Process from local directory
    local_directory = "<local folder>"  # Change this to your local directory path
    processor.process_all_local_documents(local_directory)

if __name__ == "__main__":
    main()