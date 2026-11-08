import re
from collections import Counter

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\b[\w'-]+\b")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "will",
    "with",
    "you",
    "your",
}


def _split_sentences(text: str) -> list[str]:
    """Split text on common sentence-ending punctuation and discard blank pieces."""
    return [sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(text) if sentence.strip()]


def summarize_text(text: str, summary_sentences: int) -> dict[str, str | int]:
    """Select high-scoring original sentences and return them in source order."""
    sentences = _split_sentences(text)
    words_by_sentence = [
        [
            word.lower()
            for word in _WORD.findall(sentence)
            if word.lower() not in _STOP_WORDS
        ]
        for sentence in sentences
    ]
    frequencies = Counter(word for words in words_by_sentence for word in words)

    scored_sentences = []
    for index, words in enumerate(words_by_sentence):
        score = (
            sum(frequencies[word] for word in words) / len(words)
            if words
            else 0.0
        )
        scored_sentences.append((index, score))

    selected_indices = {
        index
        for index, _ in sorted(
            scored_sentences, key=lambda item: (-item[1], item[0])
        )[:summary_sentences]
    }
    selected_sentences = [
        sentence for index, sentence in enumerate(sentences) if index in selected_indices
    ]

    return {
        "summary": " ".join(selected_sentences),
        "original_sentence_count": len(sentences),
        "summary_sentence_count": len(selected_sentences),
    }
