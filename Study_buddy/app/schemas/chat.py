"""Chat requests, responses, and validated quiz output."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ChatRequest(BaseModel):
    """A question or study action for an uploaded document."""

    document_id: int
    message: str = Field(min_length=1, max_length=4000)
    mode: Literal["auto", "qa", "summary", "quiz"] = "auto"
    quiz_count: int = Field(default=5, ge=1, le=10)


class Citation(BaseModel):
    """A source passage and its PDF page."""

    page: int
    text: str


class QuizQuestion(BaseModel):
    """A validated multiple-choice question."""

    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    answer: str
    explanation: str
    page: int = Field(gt=0)

    @model_validator(mode="after")
    def answer_is_an_option(self) -> "QuizQuestion":
        """Ensure the answer matches one of the four choices."""
        if self.answer not in self.options:
            raise ValueError("The quiz answer must match one of the options.")
        return self


class ChatResponse(BaseModel):
    """A document or web answer, optionally containing a quiz."""

    answer: str
    citations: list[Citation]
    source: Literal["document", "web"] = "document"
    mode: Literal["qa", "summary", "quiz"] = "qa"
    quiz: list[QuizQuestion] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
