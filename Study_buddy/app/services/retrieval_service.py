"""Hybrid BM25/vector retrieval followed by cross-encoder reranking."""

from functools import lru_cache
import re
from typing import Any

from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.core.logging import logger


@lru_cache(maxsize=1)
def _load_reranker() -> Any:
    """Load the configured cross-encoder once per process."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().reranker_model)


def _tokens(text: str) -> list[str]:
    """Split text into lowercase alphanumeric search terms."""
    return re.findall(r"\w+", text.lower())


def _hybrid_results(pool: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Return candidates ordered by their combined BM25/vector score."""
    return [
        {**chunk, "score": float(chunk["hybrid_score"])}
        for chunk in pool[:top_k]
    ]


def retrieve_chunks(
    question: str,
    corpus: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    top_k: int,
    use_reranker: bool = True,
) -> list[dict[str, Any]]:
    """Merge supplied BM25/vector candidates, then return reranked passages."""
    if not corpus:
        return []

    tokenized_corpus = [_tokens(str(chunk["text"])) for chunk in corpus]
    query_tokens = _tokens(question)
    if query_tokens and any(tokenized_corpus):
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_order = sorted(range(len(corpus)), key=lambda index: bm25_scores[index], reverse=True)
    else:
        bm25_order = []

    candidates: dict[str, dict[str, Any]] = {}
    for rank, chunk in enumerate(vector_results, start=1):
        key = f"{chunk['page']}:{chunk['text']}"
        candidates[key] = {**chunk, "hybrid_score": 1 / (60 + rank)}
    for rank, corpus_index in enumerate(bm25_order[: top_k * 4], start=1):
        chunk = corpus[corpus_index]
        key = f"{chunk['page']}:{chunk['text']}"
        candidate = candidates.setdefault(key, {**chunk, "hybrid_score": 0.0})
        candidate["hybrid_score"] += 1 / (60 + rank)

    pool = sorted(candidates.values(), key=lambda chunk: chunk["hybrid_score"], reverse=True)[: top_k * 4]
    if not pool:
        return []

    if not use_reranker:
        return _hybrid_results(pool, top_k)
    try:
        scores = _load_reranker().predict([[question, str(chunk["text"])] for chunk in pool])
    except RuntimeError as exc:
        if "meta tensor" not in str(exc).lower():
            raise
        logger.warning("reranker_fallback=hybrid reason=meta_tensor")
        return _hybrid_results(pool, top_k)
    ranked = sorted(zip(pool, scores), key=lambda item: float(item[1]), reverse=True)
    return [
        {**chunk, "score": float(score)}
        for chunk, score in ranked[:top_k]
    ]
