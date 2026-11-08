"""Tavily web search for questions not supported by the PDF."""

from typing import Any

from app.core.config import get_settings


def search_web(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search the web and return title, URL, and excerpt for each result."""
    api_key = get_settings().tavily_api_key
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not configured for web fallback.")
    from tavily import TavilyClient

    results: Any = TavilyClient(api_key=api_key).search(query=query, max_results=max_results)
    return [
        {
            "title": str(item.get("title", "")),
            "url": str(item["url"]),
            "content": str(item.get("content", "")),
        }
        for item in results.get("results", [])
        if item.get("url")
    ]
