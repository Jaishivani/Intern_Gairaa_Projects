"""Tests for language-model response validation and fallbacks."""

from app.services import llm_service


def test_polish_summary_falls_back_when_page_citations_change(monkeypatch, caplog) -> None:
    """Keep the original cited summary if the model drops page citations."""
    summary = "Plants use chlorophyll. [Page 1] Flowers produce seeds. [Page 2]"
    monkeypatch.setattr(llm_service, "_complete", lambda *args, **kwargs: "Plants use chlorophyll.")

    result = llm_service.polish_summary(summary)

    assert result == summary
    assert "summary_polish_discarded=page_citations_changed" in caplog.text


def test_polish_summary_uses_polished_text_when_citations_match(monkeypatch) -> None:
    """Accept model polishing when all page citations are preserved."""
    summary = "Plants use chlorophyll. [Page 1]"
    polished = "Chlorophyll helps plants. [Page 1]"
    monkeypatch.setattr(llm_service, "_complete", lambda *args, **kwargs: polished)

    assert llm_service.polish_summary(summary) == polished
