from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

class LLM:
    _llm_model = None
    _embeddings_model = None
    def __init__(self):
        if LLM._llm_model is None:
            LLM._llm_model = AzureChatOpenAI(
                model="gpt-4o",
                #azure_deployment=aoai_deployment,
                #api_version=api_version,
                temperature=0
                #max_tokens=max_tokens,
                #timeout=timeout,
                #max_retries=max_retries,
                #api_key=aoai_key,
                #azure_endpoint=aoai_endpoint
            )
        if LLM._embeddings_model is None:
            LLM._embeddings_model = AzureOpenAIEmbeddings(
                azure_deployment="text-embedding-3-large"
            )
    