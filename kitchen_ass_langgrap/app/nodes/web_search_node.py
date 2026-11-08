from app.services.tavily import search_web
from app.state import KitchenAssistantState


def web_search_node(state: KitchenAssistantState) -> KitchenAssistantState:
    results = search_web(state["search_query"])
    if not results:
        raise RuntimeError("Tavily returned no useful search results")
    return {"search_results": results}
