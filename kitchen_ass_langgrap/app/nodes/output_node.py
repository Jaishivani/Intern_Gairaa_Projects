from app.state import KitchenAssistantState


def output_node(state: KitchenAssistantState) -> KitchenAssistantState:
    answer = state.get("answer", "").strip()
    if not answer:
        raise RuntimeError("The assistant did not produce an answer")
    return {"answer": answer}
