# StudyBuddy AI

StudyBuddy AI is a local study assistant for text-based PDFs. Upload a document, ask questions with page citations, generate a summary or quiz, and explore the document's topics. When a question is not supported by the document, the answer can fall back to a web search.

## Project structure

```text
Study_buddy/
├── .vscode/
│   └── settings.json
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── auth.py
│   │       ├── chat.py
│   │       ├── documents.py
│   │       └── health.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── guardrails.py
│   │   ├── logging.py
│   │   └── security.py
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   └── models.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── nodes.py
│   │   ├── state.py
│   │   └── workflow.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── chat.py
│   │   ├── document.py
│   │   └── topic.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── cache_service.py
│   │   ├── embedding_service.py
│   │   ├── llm_service.py
│   │   ├── pdf_service.py
│   │   ├── retrieval_service.py
│   │   ├── search_service.py
│   │   ├── summary_service.py
│   │   ├── topic_service.py
│   │   └── vector_store.py
│   └── main.py
├── evaluation/
│   ├── eval_retrieval.py
│   └── test_questions.json
├── frontend/
│   └── app.py
├── tests/
│   ├── __init__.py
│   ├── test_auth.py
│   ├── test_cache.py
│   ├── test_evaluation.py
│   ├── test_guardrails.py
│   ├── test_health.py
│   ├── test_llm_service.py
│   ├── test_retrieval.py
│   ├── test_vector_store.py
│   └── test_workflow.py
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

The FastAPI backend and Streamlit frontend run as separate processes. The backend stores account and document metadata in SQLite and document vectors in ChromaDB.

## Requirements

- Python 3.11 or later (tested on Python 3.14).
- pip.
- ChromaDB 1.x (`chromadb>=1.0` in `requirements.txt`).
- A Groq API key for generated answers, summaries, and quizzes.
- A Tavily API key for web fallback.
- Internet access on first use to download the configured embedding and reranker models.

## Install and run on Windows

Open PowerShell in the project directory. Create the local environment file:

```powershell
Copy-Item .env.example .env
```

Generate a strong `SECRET_KEY` and paste the output into `.env`:

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

Get API keys from the [Groq Console](https://console.groq.com/keys) and the [Tavily dashboard](https://app.tavily.com/home). Put them in `GROQ_API_KEY` and `TAVILY_API_KEY` in `.env`. Tavily is used when a document-grounded answer is not supported and web fallback is needed.

Create and activate the virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start the API in the first terminal:

```powershell
uvicorn app.main:app --reload
```

Start the Streamlit UI in a second terminal from the project directory (activate `.venv` there too):

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run frontend/app.py
```

Open the API documentation at <http://127.0.0.1:8000/docs> and the UI at <http://localhost:8501>. The first run may take longer while the embedding and reranker models load or download.

## Settings

Create `.env` from `.env.example` and configure these variables:

| Variable | Meaning |
|---|---|
| `DATABASE_URL` | SQLAlchemy database URL for accounts and document metadata. |
| `CHROMA_PATH` | Local directory used for ChromaDB vector data. |
| `SECRET_KEY` | Secret used to sign authentication tokens; replace the example with a generated value. |
| `JWT_ALGORITHM` | Signing algorithm used for JWT access tokens. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Access-token lifetime in minutes. |
| `GROQ_API_KEY` | API key used to access Groq language models. |
| `GROQ_MODEL` | Groq model identifier used for generated study responses. |
| `TAVILY_API_KEY` | API key used for web search fallback. |
| `EMBEDDING_MODEL` | Sentence-transformer model used to embed document passages and questions. |
| `RERANKER_MODEL` | Cross-encoder model used to rerank retrieved passages. |
| `MAX_UPLOAD_SIZE_MB` | Maximum accepted PDF upload size in megabytes. |
| `CHUNK_SIZE` | Target text chunk size used when indexing PDF pages. |
| `CHUNK_OVERLAP` | Overlap between consecutive text chunks. |
| `RETRIEVAL_TOP_K` | Number of passages returned for retrieval. |
| `TOPIC_CLUSTER_COUNT` | Number of topic clusters requested for topic analysis. |
| `CACHE_TTL_SECONDS` | Lifetime of cached study responses in seconds. |
| `CACHE_MAX_ENTRIES` | Maximum number of entries in the in-memory response cache. |
| `MAX_MESSAGE_LENGTH` | Maximum allowed length of a chat message. |

## API endpoints

All endpoints except `GET /health` require `Authorization: Bearer <access_token>`. Register or log in to get a token. The API prefix is `/api/v1` for the endpoints listed below.

### Health

`GET /health` is public and returns:

```json
{"status": "healthy"}
```

### Authentication

`POST /api/v1/auth/register` creates an account. `POST /api/v1/auth/login` authenticates an existing account. Both accept:

```json
{"email": "student@example.com", "password": "a-long-password"}
```

Both return a bearer token:

```json
{"access_token": "<signed-jwt>", "token_type": "bearer"}
```

### Documents

`POST /api/v1/documents` uploads a PDF as multipart form data in a `file` field. A successful upload returns document metadata:

```json
{"id": 1, "filename": "biology.pdf", "page_count": 12, "created_at": "2026-11-04T12:00:00Z"}
```

`GET /api/v1/documents` lists the authenticated user's documents and returns an array of the same metadata objects:

```json
[{"id": 1, "filename": "biology.pdf", "page_count": 12, "created_at": "2026-11-04T12:00:00Z"}]
```

`GET /api/v1/documents/{document_id}/topics` returns topic labels and two-dimensional plotting points:

```json
{
  "topics": [{"id": 0, "label": "cell energy", "keywords": ["cell", "energy"], "chunk_count": 3}],
  "points": [{"chunk_index": 0, "page": 1, "topic_id": 0, "x": 0.1, "y": -0.2}]
}
```

`DELETE /api/v1/documents/{document_id}` deletes the document and its vector collection. It returns HTTP `204 No Content` with an empty body.

### Chat, summary, and quiz

`POST /api/v1/chat` accepts a document question or study action. `mode` can be `auto`, `qa`, `summary`, or `quiz`; `quiz_count` is optional and must be from 1 to 10.

```json
{
  "document_id": 1,
  "message": "What is the role of chlorophyll?",
  "mode": "qa",
  "quiz_count": 5
}
```

The response includes the answer, citations, source, resolved mode, quiz items, and any web URLs:

```json
{
  "answer": "Chlorophyll absorbs light energy.",
  "citations": [{"page": 2, "text": "Chlorophyll absorbs light energy..."}],
  "source": "document",
  "mode": "qa",
  "quiz": [],
  "urls": []
}
```

## Streamlit UI

Log in or create an account to open the study workspace. The sidebar lets you upload a PDF, choose one of your documents, and log out. Once a document is selected, the four tabs provide:

- **Chat:** ask questions about the selected PDF and see page citations or web sources.
- **Summary:** generate and display a document summary with citations.
- **Quiz:** generate multiple-choice questions, submit answers, and see a score and explanations.
- **Topics:** browse topic labels and view a map of passages.

## Troubleshooting

- **`chroma-hnswlib` build error:** this build issue can require Microsoft C++ Build Tools. The project uses ChromaDB 1.x (`chromadb>=1.0`) instead.
- **`DLL load failed while importing _xxhash: An Application Control policy has blocked this file`:** Windows Smart App Control or an administrator policy is blocking a compiled file. Reinstall with `pip install --force-reinstall --no-cache-dir xxhash`, or ask your administrator to allow the file.
- **`API error (500)`:** check the uvicorn terminal for the traceback first.
- **`Could not reach the API: timed out`:** the API may still be loading models on the first run, or it may not be running. Wait for startup to finish and confirm uvicorn is running.
- **PowerShell blocks virtual-environment activation:** run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` in that terminal, then activate `.venv` again.

## Tests

Run the test suite from the project directory:

```powershell
pytest -q
```

## Evaluation

`evaluation/eval_retrieval.py` compares keyword-only, vector-only, hybrid, and reranked hybrid retrieval using Precision@k, Recall@k, and MRR. Run it against an uploaded document with:

```powershell
python -m evaluation.eval_retrieval --document-id 1
```

Results to be added after running the evaluation script.

## Honest limitations

- Scanned PDFs, tables, and images in PDFs are extracted poorly. A question about a table row may not be supported by retrieved text and can fall back to the web.
- Answers depend on the language model and can be incorrect.
- The response cache is in-memory, so it is temporary and is not shared across processes.
- The judge check that decides whether an answer is supported can be wrong.
- English documents work best.
