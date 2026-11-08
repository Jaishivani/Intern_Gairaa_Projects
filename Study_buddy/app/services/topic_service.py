"""Unsupervised document topic clustering and two-dimensional projection."""

from typing import Any


def analyze_topics(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    cluster_count: int,
) -> dict[str, Any]:
    """Cluster chunk embeddings, label clusters, and return PCA plot points."""
    if len(chunks) != len(embeddings):
        raise ValueError("Chunk and embedding counts must match.")
    if not chunks:
        return {"topics": [], "points": []}

    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectors = np.asarray(embeddings)
    n_clusters = min(cluster_count, len(chunks))
    if n_clusters == 1:
        labels = np.zeros(len(chunks), dtype=int)
    else:
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(vectors)

    texts = [str(chunk["text"]) for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    try:
        matrix = vectorizer.fit_transform(texts)
        terms = vectorizer.get_feature_names_out()
    except ValueError:
        matrix = None
        terms = []

    topics: list[dict[str, Any]] = []
    for cluster in sorted(set(int(label) for label in labels)):
        member_indices = [index for index, label in enumerate(labels) if int(label) == cluster]
        keywords: list[str] = []
        if matrix is not None:
            averages = matrix[member_indices].mean(axis=0).A1
            keywords = [str(terms[index]) for index in averages.argsort()[::-1][:5]]
        topics.append({
            "id": cluster,
            "label": ", ".join(keywords) if keywords else f"Topic {cluster + 1}",
            "keywords": keywords,
            "chunk_count": len(member_indices),
        })

    if len(chunks) > 1 and vectors.shape[1] > 1:
        points = PCA(n_components=2, random_state=42).fit_transform(vectors)
    else:
        points = np.zeros((len(chunks), 2))
    return {
        "topics": topics,
        "points": [
            {
                "chunk_index": index,
                "page": int(chunk["page"]),
                "topic_id": int(labels[index]),
                "x": float(point[0]),
                "y": float(point[1]),
            }
            for index, (chunk, point) in enumerate(zip(chunks, points))
        ],
    }
