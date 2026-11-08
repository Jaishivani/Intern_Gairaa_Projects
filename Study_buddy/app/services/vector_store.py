"""Per-document ChromaDB storage and retrieval."""

from functools import lru_cache
from typing import Any

from app.core.config import get_settings


@lru_cache(maxsize=1)
def _collection_client() -> Any:
    """Create the persistent Chroma client."""
    import chromadb

    return chromadb.PersistentClient(path=get_settings().chroma_path)


def _collection_name(document_id: int) -> str:
    """Return a Chroma-compatible collection name for a document."""
    return f"document_{document_id}"


def add_document_chunks(document_id: int, texts: list[str], pages: list[int], embeddings: list[list[float]]) -> None:
    """Persist the chunks, page metadata, and embeddings for one document."""
    if not (len(texts) == len(pages) == len(embeddings)):
        raise ValueError("Chunk text, page, and embedding counts must match.")
    collection = _collection_client().get_or_create_collection(_collection_name(document_id))
    if not texts:
        return
    collection.add(
        ids=[f"chunk_{index}" for index in range(len(texts))],
        documents=texts,
        metadatas=[{"page": page} for page in pages],
        embeddings=embeddings,
    )


def query_document(document_id: int, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
    """Return the closest passages for the supplied query embedding."""
    from chromadb.errors import NotFoundError

    try:
        collection = _collection_client().get_collection(_collection_name(document_id))
    except NotFoundError:
        return []
    count = collection.count()
    if not count:
        return []
    result = collection.query(query_embeddings=[query_embedding], n_results=min(top_k, count))
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return [
        {"text": text, "page": metadata["page"], "score": distance}
        for text, metadata, distance in zip(documents, metadatas, distances)
    ]


def get_document_chunks(document_id: int) -> list[dict[str, Any]]:
    """Return all stored passages and their page metadata for one document."""
    from chromadb.errors import NotFoundError

    try:
        collection = _collection_client().get_collection(_collection_name(document_id))
    except NotFoundError:
        return []
    result = collection.get(include=["documents", "metadatas"])
    records = sorted(
        zip(result["ids"], result["documents"], result["metadatas"]),
        key=lambda record: int(record[0].rsplit("_", 1)[1]),
    )
    return [
        {"text": text, "page": metadata["page"]}
        for _, text, metadata in records
    ]


def delete_document_chunks(document_id: int) -> None:
    """Delete the vector collection for a document if it exists."""
    from chromadb.errors import NotFoundError

    name = _collection_name(document_id)
    client = _collection_client()
    try:
        client.delete_collection(name)
    except NotFoundError:
        return
