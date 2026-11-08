"""Compare retrieval strategies against expected-page question annotations."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.services.embedding_service import embed_texts
from app.services.retrieval_service import _tokens, retrieve_chunks
from app.services.vector_store import get_document_chunks


def _chunks_at_size(chunks: list[dict[str, Any]], size: int, overlap: int) -> list[dict[str, Any]]:
    """Rechunk stored page text at a requested character size."""
    pages: dict[int, list[str]] = {}
    for chunk in chunks:
        pages.setdefault(int(chunk["page"]), []).append(str(chunk["text"]))
    result: list[dict[str, Any]] = []
    for page, page_chunks in pages.items():
        text = page_chunks[0]
        for fragment in page_chunks[1:]:
            shared = min(overlap, len(text), len(fragment))
            matching_overlap = next(
                (length for length in range(shared, 0, -1) if text.endswith(fragment[:length])),
                0,
            )
            text += fragment[matching_overlap:]
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            passage = text[start:end].strip()
            if passage:
                result.append({"text": passage, "page": page})
            if end == len(text):
                break
            start = end - min(overlap, size - 1)
    return result


def _rankings(
    question: str,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    """Produce keyword, vector, hybrid, and reranked hybrid candidates."""
    tokenized = [_tokens(str(chunk["text"])) for chunk in chunks]
    query_tokens = _tokens(question)
    if query_tokens and any(tokenized):
        keyword_scores = BM25Okapi(tokenized).get_scores(query_tokens)
        keyword_order = np.argsort(keyword_scores)[::-1]
    else:
        keyword_order = np.arange(len(chunks))
    query_vector = np.asarray(embed_texts([question])[0])
    document_vectors = np.asarray(embeddings)
    vector_scores = document_vectors @ query_vector
    vector_order = np.argsort(vector_scores)[::-1]
    keyword_results = [chunks[int(index)] for index in keyword_order[:top_k]]
    vector_results = [
        {**chunks[int(index)], "score": float(vector_scores[index])}
        for index in vector_order[: min(len(chunks), top_k * 4)]
    ]
    return {
        "keyword": keyword_results,
        "vector": vector_results[:top_k],
        "hybrid": retrieve_chunks(question, chunks, vector_results, top_k, use_reranker=False),
        "hybrid + reranker": retrieve_chunks(question, chunks, vector_results, top_k),
    }


def _measure(
    ranked_results: list[list[dict[str, Any]]],
    expected_pages: list[int],
    top_k: int,
) -> tuple[float, float, float]:
    """Calculate Precision@k, Recall@k, and MRR for one method."""
    precision_values: list[float] = []
    recall_values: list[float] = []
    reciprocal_ranks: list[float] = []
    for chunks, expected_page in zip(ranked_results, expected_pages):
        relevant_ranks = [
            rank
            for rank, chunk in enumerate(chunks, start=1)
            if int(chunk["page"]) == expected_page
        ]
        precision_values.append(len(relevant_ranks) / top_k)
        recall_values.append(float(bool(relevant_ranks)))
        reciprocal_ranks.append(1 / relevant_ranks[0] if relevant_ranks else 0.0)
    count = len(expected_pages)
    return (
        sum(precision_values) / count,
        sum(recall_values) / count,
        sum(reciprocal_ranks) / count,
    )


def evaluate(document_id: int, question_file: Path, chunk_sizes: list[int], top_k: int) -> None:
    """Run retrieval comparisons for all configured chunk sizes and print metrics."""
    if top_k <= 0 or any(size <= 0 for size in chunk_sizes):
        raise ValueError("Top-k and chunk sizes must be positive integers.")
    questions = json.loads(question_file.read_text(encoding="utf-8"))
    if not questions:
        raise ValueError("The evaluation question file must contain at least one question.")
    stored_chunks = get_document_chunks(document_id)
    if not stored_chunks:
        raise ValueError(f"No indexed chunks found for document {document_id}.")

    methods = ["keyword", "vector", "hybrid", "hybrid + reranker"]
    print("| Chunk size | Method | Precision@k | Recall@k | MRR |")
    print("|---:|---|---:|---:|---:|")
    for chunk_size in chunk_sizes:
        chunks = _chunks_at_size(stored_chunks, chunk_size, get_settings().chunk_overlap)
        embeddings = embed_texts([str(chunk["text"]) for chunk in chunks])
        all_results: dict[str, list[list[dict[str, Any]]]] = {method: [] for method in methods}
        for item in questions:
            results = _rankings(str(item["question"]), chunks, embeddings, top_k)
            for method in methods:
                all_results[method].append(results[method])
        expected_pages = [int(item["expected_page"]) for item in questions]
        for method in methods:
            precision, recall, mrr = _measure(all_results[method], expected_pages, top_k)
            print(
                f"| {chunk_size} | {method} | {precision:.3f} | {recall:.3f} | {mrr:.3f} |"
            )


def main() -> None:
    """Parse evaluation options and run the retrieval benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-id", type=int, required=True)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).with_name("test_questions.json"),
    )
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[300, 800])
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.document_id, args.questions, args.chunk_sizes, args.top_k)


if __name__ == "__main__":
    main()
