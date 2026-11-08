"""Document API schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.schemas.topic import TopicAnalysis


class DocumentResponse(BaseModel):
    """Public document metadata."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    page_count: int
    created_at: datetime


class DocumentTopicsResponse(TopicAnalysis):
    """Topic analysis response associated with a document."""
