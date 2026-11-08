"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.routes import auth, chat, documents, health
from app.core.config import get_settings
from app.db.database import initialize_database

initialize_database()
app = FastAPI(title=get_settings().app_name)
app.include_router(health.router)
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(documents.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
