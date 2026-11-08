from fastapi import APIRouter

from app.schemas.summary import SummaryRequest, SummaryResponse
from app.services.summarizer_service import summarize_text

router = APIRouter(prefix="/api/v1", tags=["summarization"])


@router.post("/summarize", response_model=SummaryResponse)
def summarize(request: SummaryRequest) -> SummaryResponse:
    """Return an extractive summary of the submitted text."""
    result = summarize_text(request.text, request.summary_sentences)
    return SummaryResponse(**result)
