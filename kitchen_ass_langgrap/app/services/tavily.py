import requests

from app.config import get_required_setting

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def search_web(query: str) -> list[dict[str, str]]:
    response = requests.post(
        TAVILY_SEARCH_URL,
        json={
            "api_key": get_required_setting("TAVILY_API_KEY"),
            "query": query,
            "search_depth": "advanced",
            "max_results": 5,
            "include_answer": False,
        },
        timeout=30,
    )
    try:
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as error:
        raise RuntimeError("Tavily web search failed") from error

    results = payload.get("results", [])
    return [
        {
            "title": str(item.get("title", "")),
            "url": str(item.get("url", "")),
            "content": str(item.get("content", "")),
        }
        for item in results
        if item.get("title") and item.get("url") and item.get("content")
    ]
