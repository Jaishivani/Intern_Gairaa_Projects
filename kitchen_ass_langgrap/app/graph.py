from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.nodes.input_node import input_node
from app.nodes.llm_node import llm_node
from app.nodes.output_node import output_node
from app.nodes.web_search_node import web_search_node
from app.state import KitchenAssistantState


def route_after_llm(state: KitchenAssistantState) -> Literal["web_search", "output"]:
    return "web_search" if state.get("search_required", False) else "output"


workflow = StateGraph(KitchenAssistantState)
workflow.add_node("input", input_node)
workflow.add_node("llm", llm_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("output", output_node)
workflow.add_edge(START, "input")
workflow.add_edge("input", "llm")
workflow.add_conditional_edges("llm", route_after_llm)
workflow.add_edge("web_search", "llm")
workflow.add_edge("output", END)

graph = workflow.compile()
