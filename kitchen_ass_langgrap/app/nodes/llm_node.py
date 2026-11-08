from app.services.nvidia import answer_with_search_results, classify_question
from app.state import KitchenAssistantState


def llm_node(state: KitchenAssistantState) -> KitchenAssistantState:
    question = state["question"]
    if state.get("search_results"):
        answer = answer_with_search_results(question, state["search_results"])
        sources = "\n\nSources:\n" + "\n".join(
            f"[{index}. {result['title']}]({result['url']})"
            for index, result in enumerate(state["search_results"], start=1)
        )
        return {
            "search_required": False,
            "answer": answer + sources,
        }

    decision = classify_question(question)
    if decision["search_required"]:
        return {
            "search_required": True,
            "search_query": str(decision["search_query"]).strip(),
        }
    return {"search_required": False, "answer": str(decision["answer"]).strip()}
