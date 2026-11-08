"""SQLAlchemy engine, session factory, and dependency."""

from collections.abc import Generator

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.core.config import get_settings

_settings = get_settings()
_connect_args = {"check_same_thread": False} if _settings.database_url.startswith("sqlite") else {}
engine = create_engine(_settings.database_url, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and always close it."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def initialize_database(database_engine: Engine = engine) -> None:
    """Create tables and add the nullable owner column to pre-auth databases."""
    from app.db import models

    models.User.__table__.metadata.create_all(bind=database_engine)
    columns = {column["name"] for column in inspect(database_engine).get_columns("documents")}
    if "owner_id" not in columns:
        with database_engine.begin() as connection:
            connection.execute(
                text("ALTER TABLE documents ADD COLUMN owner_id INTEGER REFERENCES users(id)")
            )
