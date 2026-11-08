"""Study workflow and document chat-history handling."""

import time

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ChatMessage
from app.graph.nodes import (
    answer_node,
    cache_check_node,
    finalize_node,
    guardrail_node,
    judge_node,
    quiz_node,
    retrieve_node,
    route_node,
    summary_node,
    web_search_node,
)
from app.graph.state import StudyState


def run_question_answer(
    document_id: int,
    message: str,
    db: Session,
    mode: str = "auto",
    quiz_count: int = 5,
    user_id: int | None = None,
) -> StudyState:
    """Run a study request and persist its exchange for future context."""
    history_query = select(ChatMessage).where(ChatMessage.document_id == document_id)
    if user_id is not None:
        history_query = history_query.where(ChatMessage.document.has(owner_id=user_id))
    recent = db.scalars(
        history_query.order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc()).limit(2)
    ).all()
    history = [
        entry
        for chat_message in reversed(recent)
        for entry in (
            {"role": "user", "content": chat_message.question},
            {"role": "assistant", "content": chat_message.answer},
        )
    ]

    state: StudyState = {
        "document_id": document_id,
        "message": message,
        "requested_mode": mode,
        "quiz_count": quiz_count,
        "history": history,
        "started_at": time.perf_counter(),
        "user_id": user_id or 0,
    }

    state.update(guardrail_node(state))
    if not state.get("blocked", False):
        state.update(route_node(state))
        state.update(cache_check_node(state))
        if not state.get("cache_hit", False):
            if state["mode"] == "summary":
                state.update(summary_node(state))
            else:
                state.update(retrieve_node(state))
                if state["mode"] == "quiz":
                    state.update(quiz_node(state))
                else:
                    state.update(answer_node(state))
                    state.update(judge_node(state))
                    if not state.get("supported", False):
                        state.update(web_search_node(state))

    result = finalize_node(state)
    db.add(ChatMessage(document_id=document_id, question=message, answer=result["answer"]))
    db.commit()
    return result
