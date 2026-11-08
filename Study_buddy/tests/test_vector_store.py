"""Tests for missing ChromaDB collections."""

import pytest

from app.services import vector_store


def test_missing_collection_returns_no_chunks(monkeypatch) -> None:
    """A missing Chroma collection is treated as an empty document index."""
    from chromadb.errors import NotFoundError

    class MissingCollectionClient:
        def get_collection(self, name):
            raise NotFoundError("Collection does not exist")

    monkeypatch.setattr(vector_store, "_collection_client", lambda: MissingCollectionClient())

    assert vector_store.query_document(123, [0.1, 0.2], 5) == []
    assert vector_store.get_document_chunks(123) == []


def test_delete_missing_collection_is_safe(monkeypatch) -> None:
    """Deleting a document without an index is an idempotent no-op."""
    from chromadb.errors import NotFoundError

    class MissingCollectionClient:
        def delete_collection(self, name):
            raise NotFoundError("Collection does not exist")

    monkeypatch.setattr(vector_store, "_collection_client", lambda: MissingCollectionClient())

    vector_store.delete_document_chunks(123)


def test_unexpected_collection_error_is_not_silenced(monkeypatch) -> None:
    """Storage failures are surfaced instead of reported as an empty index."""
    class BrokenCollectionClient:
        def get_collection(self, name):
            raise OSError("storage unavailable")

    monkeypatch.setattr(vector_store, "_collection_client", lambda: BrokenCollectionClient())

    with pytest.raises(OSError, match="storage unavailable"):
        vector_store.get_document_chunks(123)
