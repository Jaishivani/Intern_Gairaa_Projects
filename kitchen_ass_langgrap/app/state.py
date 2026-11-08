from typing import TypedDict


class KitchenAssistantState(TypedDict, total=False):
    question: str
    search_required: bool
    search_query: str
    search_results: list[dict[str, str]]
    answer: str
