import json
from typing import Optional, List
from tavily import AsyncTavilyClient
from pydantic import BaseModel
from langchain_core.tools import tool

class SearchEntry(BaseModel):
    title: str
    url: str
    content: str

@tool
async def tavily_search(query: str, max_results: Optional[int] = None) -> List[SearchEntry]:
    r"""Use Tavily Search API to search information for the given query. 

    Args:
        query (str): The query to be searched.
        max_results (Optional[int]): The maximum number of search results to return.

    Returns:
        List[SearchEntry]: A list of search results. Each entry contains title, url, and content.
    """
    client = AsyncTavilyClient()

    try:
        results = await client.search(query, max_results=max_results) # type: ignore
        formed_results = [
            SearchEntry(
                title=result.get("title", ""),
                url=result.get("url", ""),
                content=result.get("content", "")
            ) for result in results['results']
        ]
        return formed_results
    except Exception as e:
        return []
