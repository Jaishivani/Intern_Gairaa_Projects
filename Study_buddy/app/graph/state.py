"""State passed between nodes in the study workflow."""

from typing import TypedDict

from app.schemas.chat import QuizQuestion


class StudyState(TypedDict, total=False):
    """Inputs, retrieved evidence, and outputs for one study request."""

    document_id: int
    message: str
    requested_mode: str
    mode: str
    quiz_count: int
    history: list[dict[str, str]]
    chunks: list[dict[str, object]]
    answer: str
    source: str
    citations: list[dict[str, object]]
    quiz: list[QuizQuestion]
    urls: list[str]
    supported: bool
    blocked: bool
    cache_hit: bool
    cached_response: dict[str, object]
    started_at: float
    user_id: int
