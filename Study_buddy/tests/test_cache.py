"""Response cache tests."""

from app.services import cache_service


def test_cache_normalizes_question_and_separates_modes(monkeypatch) -> None:
    """Whitespace/case variants hit the same key, but different modes do not."""
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "cache_ttl_seconds", 60)
    cache_service.clear_cache()
    cache_service.store_response(3, "  What   is ATP? ", "qa", {"answer": "Energy carrier."})

    assert cache_service.get_cached_response(3, "what is atp?", "qa") == {
        "answer": "Energy carrier."
    }
    assert cache_service.get_cached_response(3, "what is atp?", "summary") is None


def test_cache_expiration(monkeypatch) -> None:
    """Expired responses are removed rather than returned."""
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "cache_ttl_seconds", 1)
    monkeypatch.setattr(cache_service.time, "monotonic", lambda: 10.0)
    cache_service.clear_cache()
    cache_service.store_response(5, "What is RNA?", "qa", {"answer": "Genetic material"})
    monkeypatch.setattr(cache_service.time, "monotonic", lambda: 12.0)

    assert cache_service.get_cached_response(5, "What is RNA?", "qa") is None


def test_cache_evicts_old_entries_when_full(monkeypatch) -> None:
    """The cache stays under its configured maximum entry count."""
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "cache_ttl_seconds", 60)
    monkeypatch.setattr(get_settings(), "cache_max_entries", 1)
    cache_service.clear_cache()
    cache_service.store_response(1, "First question?", "qa", {"answer": "first"})
    cache_service.store_response(2, "Second question?", "qa", {"answer": "second"})

    assert cache_service.get_cached_response(1, "First question?", "qa") is None
    assert cache_service.get_cached_response(2, "Second question?", "qa") == {"answer": "second"}


def test_document_cache_can_be_invalidated() -> None:
    """Deleting one document clears only that document's cached answers."""
    cache_service.clear_cache()
    cache_service.store_response(1, "Question one?", "qa", {"answer": "one"})
    cache_service.store_response(2, "Question two?", "qa", {"answer": "two"})
    cache_service.clear_document_cache(1)

    assert cache_service.get_cached_response(1, "Question one?", "qa") is None
    assert cache_service.get_cached_response(2, "Question two?", "qa") == {"answer": "two"}
