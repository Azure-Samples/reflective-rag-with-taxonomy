CONSOLIDATION_PROMPT = """Create a comprehensive answer to the user's question by synthesizing the research conducted on different taxonomies.

User Question: {question}

Research Results:
{research_results}

Generate a clear, coherent answer that integrates information from all taxonomies and provides a complete response to the user's question. 
Ensure the answer is well-structured and flows naturally between different aspects of the research.
Cite specific results when appropriate.
Do not include any information about the research process or taxonomies in your answer.
"""