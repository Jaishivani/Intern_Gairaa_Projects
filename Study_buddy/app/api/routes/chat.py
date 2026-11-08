"""Document question-answer endpoint."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.database import get_db
from app.db.models import Document, User
from app.graph.workflow import run_question_answer
from app.schemas.chat import ChatRequest, ChatResponse, Citation

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    """Run a study request for an uploaded document."""
    document = db.scalar(
        select(Document).where(
            Document.id == request.document_id,
            Document.owner_id == current_user.id,
        )
    )
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    try:
        state = run_question_answer(
            request.document_id,
            request.message,
            db,
            request.mode,
            request.quiz_count,
            current_user.id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    citations = [
        Citation(page=int(chunk["page"]), text=str(chunk["text"]))
        for chunk in state.get("citations", [])
    ]
    return ChatResponse(
        answer=state["answer"],
        citations=citations,
        source=state.get("source", "document"),
        mode=state["mode"],
        quiz=state.get("quiz", []),
        urls=state.get("urls", []),
    )
