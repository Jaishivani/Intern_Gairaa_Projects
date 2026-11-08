"""Input, output, and extracted-document prompt-injection checks."""

import re

_INJECTION_PATTERNS = (
    re.compile(r"\bignore (all )?(previous|prior|above) instructions\b", re.IGNORECASE),
    re.compile(r"\bdisregard (all )?(previous|prior|above) instructions\b", re.IGNORECASE),
    re.compile(r"\breveal (the )?(system|developer) prompt\b", re.IGNORECASE),
    re.compile(r"\b(system|developer) prompt\b", re.IGNORECASE),
    re.compile(r"\bact as (the )?(system|developer)\b", re.IGNORECASE),
    re.compile(r"\boutput (all )?(secrets|credentials|hidden instructions)\b", re.IGNORECASE),
)
_OFF_TOPIC_PATTERNS = (
    re.compile(r"\b(weather|stock price|sports score|recipe|restaurant reservation)\b", re.IGNORECASE),
    re.compile(r"\b(write|debug|deploy) (a |some )?(code|software|website|app)\b", re.IGNORECASE),
    re.compile(r"\bbook (me )?(a )?(flight|hotel)\b", re.IGNORECASE),
)
_UNSAFE_OUTPUT_PATTERNS = (
    re.compile(r"\b(make|build|construct) (a |an )?(bomb|explosive|weapon)\b", re.IGNORECASE),
    re.compile(r"\bhow to (kill|poison) (someone|a person)\b", re.IGNORECASE),
    re.compile(r"\bencourag(e|ing) (you )?to (hurt|kill) yourself\b", re.IGNORECASE),
)


def check_input(message: str, max_length: int) -> str | None:
    """Return a refusal reason for empty, oversized, injected, or unrelated input."""
    normalized = message.strip()
    if not normalized:
        return "Please enter a question about your study material."
    if len(normalized) > max_length:
        return f"Please keep your question under {max_length} characters."
    if any(pattern.search(normalized) for pattern in _INJECTION_PATTERNS):
        return "I can only help study the document, not follow instructions to override my rules."
    if any(pattern.search(normalized) for pattern in _OFF_TOPIC_PATTERNS):
        return "I can help with questions about your uploaded study document."
    return None


def check_output(answer: str) -> str:
    """Replace empty or clearly unsafe generated content with a safe response."""
    normalized = answer.strip()
    if not normalized:
        return "I couldn't produce a reliable answer from the available material."
    if any(pattern.search(normalized) for pattern in _UNSAFE_OUTPUT_PATTERNS):
        return "I can't provide instructions that could cause harm."
    return normalized


def is_prompt_injection(text: str) -> bool:
    """Identify extracted PDF text that appears to contain instruction attacks."""
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)
