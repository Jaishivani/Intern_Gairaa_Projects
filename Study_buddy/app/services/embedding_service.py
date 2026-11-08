"""Pretrained sentence-transformer embedding loader."""

from functools import lru_cache
from typing import Any

from app.core.config import get_settings


@lru_cache(maxsize=1)
def _load_model() -> Any:
    """Load the configured embedding model once per process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode text passages as normalized embedding vectors."""
    if not texts:
        return []
    vectors = _load_model().encode(texts, normalize_embeddings=True)
    return vectors.tolist()
