# Bank Account Chatbot

This is a small educational FastAPI project that demonstrates a banking chatbot using:

- FastAPI
- Groq LLM
- Excel-based data storage
- Input/output guardrails
- Simple token-based login
- Docker support

## Features

- User login with a demo username/password pair
- Protected account and transaction endpoints
- Banking-scope input guardrails
- Sensitive information blocking
- Safe context sent to the LLM
- Lightweight Docker setup

## Local setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create the Excel dataset:

   ```bash
   python create_data.py
   ```

4. Copy `.env.example` to `.env` and add your Groq API key:

   ```bash
   copy .env.example .env
   ```

   Then update `.env` with:

   ```env
   GROQ_API_KEY=your_groq_api_key
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

5. Run the API:

   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. Open the API docs at:

   ```text
   http://localhost:8000/docs
   ```

## Demo login

Use one of the dummy users:

```json
{
  "username": "shivani",
  "password": "pass123"
}
```

The dataset stores a SHA-256 password hash, so the app can validate the login without returning the password.

## Password handling

This project stores password hashes in the Excel dataset rather than plain text. The hash is generated using a simple SHA-256 approach for demonstration purposes. This is intentionally lightweight and is not suitable for production-grade security.

## Important security notes

- Passwords, PINs, CVV, OTPs, tokens, and API keys are never returned by the API.
- The LLM only receives safe account information.
- Sensitive questions are blocked before the request reaches Groq.
- The token store is in-memory and intended only for local demonstration.

## Docker

Build and run with Docker Compose:

```bash
docker-compose up --build
```

The app will be available at:

```text
http://localhost:8000
```

## Notes

This is a demo project only. It is not connected to a real bank and is not production authentication.
