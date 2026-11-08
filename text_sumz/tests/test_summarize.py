from fastapi.testclient import TestClient

from app.core.config import MAX_INPUT_LENGTH, MAX_SUMMARY_SENTENCES
from app.main import app

client = TestClient(app)


def test_summarize_returns_high_scoring_sentences_in_original_order() -> None:
    text = (
        "Oranges are rich in vitamin C. "
        "Vitamin C supports immune health. "
        "The ocean covers much of Earth. "
        "Many people enjoy reading novels."
    )

    response = client.post(
        "/api/v1/summarize",
        json={"text": text, "summary_sentences": 2},
    )

    assert response.status_code == 200
    assert response.json() == {
        "summary": "Oranges are rich in vitamin C. Vitamin C supports immune health.",
        "original_sentence_count": 4,
        "summary_sentence_count": 2,
    }


def test_short_text_is_returned_without_inventing_sentences() -> None:
    response = client.post(
        "/api/v1/summarize",
        json={"text": "A short sentence.", "summary_sentences": 3},
    )

    assert response.status_code == 200
    assert response.json() == {
        "summary": "A short sentence.",
        "original_sentence_count": 1,
        "summary_sentence_count": 1,
    }


def test_empty_text_is_rejected() -> None:
    response = client.post("/api/v1/summarize", json={"text": ""})

    assert response.status_code == 422
    assert "at least 1 character" in response.text


def test_whitespace_only_text_is_rejected() -> None:
    response = client.post("/api/v1/summarize", json={"text": "   \n\t"})

    assert response.status_code == 422
    assert "non-whitespace" in response.text


def test_input_over_maximum_length_is_rejected() -> None:
    response = client.post(
        "/api/v1/summarize",
        json={"text": "x" * (MAX_INPUT_LENGTH + 1)},
    )

    assert response.status_code == 422


def test_summary_sentence_count_must_be_in_range() -> None:
    for requested_count in (0, MAX_SUMMARY_SENTENCES + 1):
        response = client.post(
            "/api/v1/summarize",
            json={"text": "A valid sentence.", "summary_sentences": requested_count},
        )
        assert response.status_code == 422


def test_non_string_text_is_rejected() -> None:
    response = client.post("/api/v1/summarize", json={"text": 123})

    assert response.status_code == 422


def test_missing_text_is_rejected() -> None:
    response = client.post("/api/v1/summarize", json={})

    assert response.status_code == 422
