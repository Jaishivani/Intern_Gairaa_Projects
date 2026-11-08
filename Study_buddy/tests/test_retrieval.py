"""Hybrid retrieval tests."""

from app.services.retrieval_service import retrieve_chunks


def test_hybrid_retrieval_merges_searches_and_reranks(monkeypatch) -> None:
    """BM25-only candidates can join vector results and win reranking."""
    from app.services import retrieval_service

    chunks = [
        {"text": "Ocean water contains salt.", "page": 1},
        {"text": "Chlorophyll helps plants absorb light.", "page": 2},
    ]
    vector_result = [{"text": chunks[0]["text"], "page": 1, "score": 0.1}]

    class Reranker:
        def predict(self, pairs):
            return [0.2 if "Ocean" in pair[1] else 0.9 for pair in pairs]

    monkeypatch.setattr(retrieval_service, "_load_reranker", lambda: Reranker())

    results = retrieve_chunks("What role does chlorophyll play?", chunks, vector_result, 2)

    assert [chunk["page"] for chunk in results] == [2, 1]
    assert results[0]["score"] == 0.9


def test_retrieval_handles_non_alphanumeric_passages(monkeypatch) -> None:
    """Empty BM25 token sets do not break vector-based retrieval."""
    from app.services import retrieval_service

    class Reranker:
        def predict(self, pairs):
            return [0.5 for _ in pairs]

    monkeypatch.setattr(retrieval_service, "_load_reranker", lambda: Reranker())
    chunks = [{"text": "... ---", "page": 1}]
    assert retrieve_chunks("???", chunks, chunks, 1)[0]["page"] == 1


def test_retrieval_falls_back_when_reranker_has_meta_tensor_error(monkeypatch, caplog) -> None:
    """Use hybrid ranking when PyTorch cannot load the reranker weights."""
    from app.services import retrieval_service

    class BrokenReranker:
        def predict(self, pairs):
            raise RuntimeError(
                "Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty()"
            )

    monkeypatch.setattr(retrieval_service, "_load_reranker", lambda: BrokenReranker())
    corpus = [
        {"text": "Question about chlorophyll.", "page": 1},
        {"text": "Question about animal cells.", "page": 2},
    ]

    results = retrieval_service.retrieve_chunks(
        "Where is chlorophyll?", corpus, [], top_k=2
    )

    assert results[0]["page"] == 1
    assert all("score" in chunk for chunk in results)
    assert "reranker_fallback=hybrid" in caplog.text
