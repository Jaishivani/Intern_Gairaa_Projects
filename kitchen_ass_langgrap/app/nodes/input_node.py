from app.state import KitchenAssistantState


def input_node(state: KitchenAssistantState) -> KitchenAssistantState:
    question = state.get("question", "").strip()
    if not question:
        raise ValueError("Please enter a kitchen-related question.")
    return {"question": question}
