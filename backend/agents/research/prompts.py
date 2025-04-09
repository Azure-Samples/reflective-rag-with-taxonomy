QUERY_PROMPT = """Generate an effective search query for retrieving information on a specific taxonomy related to the user's question.

User Question: {question}
Taxonomy: {taxonomy}

Your task is to create a targeted search query and optional filter that will retrieve documents specifically relevant to this taxonomy.

Search queries should:
1. Be specific and focused on the taxonomy
2. Include key terms and concepts relevant to the taxonomy
3. Be concise but descriptive
4. Be tailored to retrieve factual information

If there have been previous search attempts, review them and adjust your strategy accordingly.

Previous Search Attempts:
{search_history}

Return:
1. search_query: Your optimized search query
2. filter: An optional filter to narrow results (can be None)

DEV MODE: You are in dev mode. Filter is always an empty string.
"""

REVIEW_PROMPT = """Review these search results and determine which contain relevant information for answering the user's question within the specific taxonomy.

User Question: {question}
Taxonomy: {taxonomy}

Your task is to evaluate each search result and determine if it contains information that helps address this taxonomy of the user's question.

Consider:
1. Does the result contain information relevant to this specific taxonomy?
2. Does the information help answer the user's question from the perspective of this taxonomy?
3. Is the information factual and reliable?

Respond with:
1. thought_process: Your analysis of each result
2. valid_results: List of indices (0-N) for useful results
3. invalid_results: List of indices (0-N) for irrelevant results
4. decision: Either "retry" if we need more info or "finalize" if we have sufficient information

Current Search Results:
{current_results}

Previously Vetted Results:
{vetted_results}

Search History:
{search_history}
"""
