# AI Text Summarizer API

A small REST API built with FastAPI. It creates extractive summaries by scoring
sentences with the frequencies of their non-stop words, selecting the highest
scoring sentences, and returning them in their original order. It does not use
model training, a dataset, a paid API, or a pretrained language model.

## Project structure

```text
AI_Text_Summarizer/
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py
│   │       └── summarize.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── summary.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── summarizer_service.py
│   └── main.py
├── tests/
│   ├── __init__.py
│   ├── test_health.py
│   └── test_summarize.py
├── requirements.txt
├── .gitignore
└── README.md
```

The `__init__.py` files mark the application directories as Python packages.

## Requirements

- Python 3.11 or later
- pip

## Install and run (Windows Command Prompt)

Run these commands from the project directory:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open Swagger UI at <http://127.0.0.1:8000/docs>. The OpenAPI schema is at
<http://127.0.0.1:8000/openapi.json>.

The API limits input to 10,000 characters and requested summaries to 1–10
sentences. Change `MAX_INPUT_LENGTH`, `DEFAULT_SUMMARY_SENTENCES`, or
`MAX_SUMMARY_SENTENCES` in `app/core/config.py` to adjust those settings.

## Endpoints

### `GET /health`

Example response:

```json
{
  "status": "healthy"
}
```

### `POST /api/v1/summarize`

`text` is required and cannot be empty or whitespace-only.
`summary_sentences` is optional; it defaults to 3 and must be from 1 to 10.
If fewer sentences are present in the text than requested, all available
sentences are returned.

Example request:

```json
{
  "text": "Artificial intelligence is transforming many industries. Healthcare organizations use AI to assist with medical image analysis. Businesses also use AI to automate repetitive tasks. Responsible implementation remains important.",
  "summary_sentences": 2
}
```

Example response shape:

```json
{
  "summary": "Healthcare organizations use AI to assist with medical image analysis. Businesses also use AI to automate repetitive tasks.",
  "original_sentence_count": 4,
  "summary_sentence_count": 2
}
```

The summary text is selected at request time by the summarization service; this
example illustrates the response format.

Invalid JSON fields, blank text, text longer than the configured maximum, and
summary sentence counts outside the supported range receive HTTP `422` with
FastAPI's JSON validation details.

## Try it in Swagger UI

1. Start the server and open <http://127.0.0.1:8000/docs>.
2. Expand `POST /api/v1/summarize` and choose **Try it out**.
3. Enter a JSON body with `text` and optionally `summary_sentences`.
4. Choose **Execute** and inspect the HTTP status and JSON response.
5. Use `GET /health` in the same way to check the running service.

## Tests

With the virtual environment active, run:

```bat
pytest -q
```

The API tests use FastAPI's in-process test client and need no external service.

## Modules in brief

- `app/main.py` creates the FastAPI app, adds API metadata, and registers routers.
- `app/api/routes/summarize.py` accepts a validated request and calls the service.
- `app/api/routes/health.py` provides the health-check response.
- `app/core/config.py` holds the length and default settings.
- `app/schemas/summary.py` defines and validates the request and response shapes.
- `app/services/summarizer_service.py` handles sentence splitting and scoring,
  separately from HTTP concerns.
- `tests/` checks summarization, invalid requests, and the health endpoint.

## Algorithm and limitations

This is an extractive, not generative, summarizer. It splits on common `.`, `!`,
and `?` sentence endings, counts non-stop-word frequencies across the text, and
scores each sentence by the average frequency of its words. It selects the
highest-scoring sentences and restores their original order and wording.

This lightweight heuristic does not understand context, paraphrase, or
guarantee factual correctness. Sentence splitting is intentionally simple, so
abbreviations and unusual punctuation can be handled imperfectly. Requested
summary length is a maximum; the result may contain fewer sentences if the
input is shorter.
