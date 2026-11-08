"""Small thread-safe in-memory cache for document study responses."""

from threading import RLock
import time
from typing import Any

from app.core.config import get_settings

_entries: dict[tuple[int, str, str, int], tuple[float, dict[str, Any]]] = {}
_lock = RLock()


def _key(document_id: int, question: str, mode: str, quiz_count: int) -> tuple[int, str, str, int]:
    """Normalize a question and include response mode in its cache identity."""
    normalized = " ".join(question.casefold().split())
    return document_id, normalized, mode, quiz_count if mode == "quiz" else 0


def get_cached_response(
    document_id: int,
    question: str,
    mode: str,
    quiz_count: int = 0,
) -> dict[str, Any] | None:
    """Return an unexpired response copy, removing stale entries."""
    key = _key(document_id, question, mode, quiz_count)
    now = time.monotonic()
    with _lock:
        entry = _entries.get(key)
        if entry is None:
            return None
        expires_at, response = entry
        if expires_at <= now:
            del _entries[key]
            return None
        return dict(response)


def store_response(
    document_id: int,
    question: str,
    mode: str,
    response: dict[str, Any],
    quiz_count: int = 0,
) -> None:
    """Store a copy of the response until its configured expiry time."""
    key = _key(document_id, question, mode, quiz_count)
    settings = get_settings()
    now = time.monotonic()
    expires_at = now + settings.cache_ttl_seconds
    with _lock:
        expired_keys = [entry_key for entry_key, (expiry, _) in _entries.items() if expiry <= now]
        for expired_key in expired_keys:
            del _entries[expired_key]
        while len(_entries) >= settings.cache_max_entries and key not in _entries:
            oldest_key = min(_entries, key=lambda entry_key: _entries[entry_key][0])
            del _entries[oldest_key]
        _entries[key] = (expires_at, dict(response))


def clear_cache() -> None:
    """Clear all cached responses, primarily for isolated tests."""
    with _lock:
        _entries.clear()


def clear_document_cache(document_id: int) -> None:
    """Remove cached responses associated with one deleted document."""
    with _lock:
        stale_keys = [key for key in _entries if key[0] == document_id]
        for stale_key in stale_keys:
            del _entries[stale_key]
