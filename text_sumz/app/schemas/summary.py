from pydantic import BaseModel, Field, field_validator

from app.core.config import (
    DEFAULT_SUMMARY_SENTENCES,
    MAX_INPUT_LENGTH,
    MAX_SUMMARY_SENTENCES,
)


class SummaryRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=MAX_INPUT_LENGTH,
        description=f"Text to summarize (maximum {MAX_INPUT_LENGTH} characters).",
    )
    summary_sentences: int = Field(
        default=DEFAULT_SUMMARY_SENTENCES,
        ge=1,
        le=MAX_SUMMARY_SENTENCES,
        description=(
            "Maximum number of sentences to return "
            f"(between 1 and {MAX_SUMMARY_SENTENCES})."
        ),
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must contain at least one non-whitespace character.")
        return value


class SummaryResponse(BaseModel):
    summary: str = Field(description="Extracted sentences in their original order.")
    original_sentence_count: int = Field(
        ge=1, description="Number of sentences detected in the input."
    )
    summary_sentence_count: int = Field(
        ge=1, description="Number of sentences in the returned summary."
    )
