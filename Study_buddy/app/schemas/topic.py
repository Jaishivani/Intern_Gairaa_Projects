"""Document topic analysis response schemas."""

from pydantic import BaseModel


class Topic(BaseModel):
    """A cluster label and its representative keywords."""

    id: int
    label: str
    keywords: list[str]
    chunk_count: int


class TopicPoint(BaseModel):
    """A document chunk projected into two-dimensional topic space."""

    chunk_index: int
    page: int
    topic_id: int
    x: float
    y: float


class TopicAnalysis(BaseModel):
    """Topic clusters and plot coordinates for a document."""

    topics: list[Topic]
    points: list[TopicPoint]
