"""Extractive summaries of page-aware document chunks."""

import re
from typing import Any


def extractive_summary(
    chunks: list[dict[str, Any]],
    sentence_count: int = 5,
) -> tuple[str, list[dict[str, Any]]]:
    """Select representative sentences with TF-IDF scoring and retain pages."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    sentences: list[dict[str, Any]] = []
    for chunk in chunks:
        for text in re.split(r"(?<=[.!?])\s+", str(chunk["text"]).strip()):
            if text:
                sentences.append({"text": text, "page": int(chunk["page"])})
    if not sentences:
        return "", []

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform([str(sentence["text"]) for sentence in sentences])
        scores = matrix.sum(axis=1).A1
    except ValueError:
        scores = [len(str(sentence["text"])) for sentence in sentences]

    selected_indices = sorted(
        sorted(range(len(sentences)), key=lambda index: float(scores[index]), reverse=True)[:sentence_count]
    )
    selected = [sentences[index] for index in selected_indices]
    summary = " ".join(f"{item['text']} [Page {item['page']}]" for item in selected)
    return summary, selected
