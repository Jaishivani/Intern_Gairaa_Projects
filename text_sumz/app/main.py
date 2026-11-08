from fastapi import FastAPI

from app.api.routes import health, summarize

app = FastAPI(
    title="AI Text Summarizer API",
    description=(
        "Creates concise summaries by selecting important sentences from the "
        "original text using word-frequency scoring. It does not use a trained "
        "generative model."
    ),
    version="1.0.0",
)

app.include_router(health.router)
app.include_router(summarize.router)
