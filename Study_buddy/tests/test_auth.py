"""Authentication and document-ownership API tests."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, create_engine, inspect
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.database import Base, get_db, initialize_database
from app.db.models import Document, User
from app.main import app


@pytest.fixture
def client():
    """Provide an isolated database and API client."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)

    def override_db():
        yield session

    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as test_client:
        yield test_client, session
    app.dependency_overrides.clear()
    session.close()
    engine.dispose()


def _register(client: TestClient, email: str) -> str:
    """Register an account and return its bearer token."""
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "secure-password-123"},
    )
    assert response.status_code == 201
    return response.json()["access_token"]


def test_register_login_and_reject_wrong_password(client) -> None:
    """Accounts can register and log in, while bad credentials are rejected."""
    test_client, _ = client
    token = _register(test_client, "Student@Example.com")
    assert token

    login = test_client.post(
        "/api/v1/auth/login",
        json={"email": "student@example.com", "password": "secure-password-123"},
    )
    assert login.status_code == 200
    assert login.json()["access_token"]

    rejected = test_client.post(
        "/api/v1/auth/login",
        json={"email": "student@example.com", "password": "wrong-password"},
    )
    assert rejected.status_code == 401


def test_documents_require_auth_and_are_scoped_to_the_owner(client) -> None:
    """Unauthenticated requests are denied and users see only their documents."""
    test_client, db = client
    first_token = _register(test_client, "first@example.com")
    second_token = _register(test_client, "second@example.com")
    first_user_id = db.query(User).filter_by(email="first@example.com").one().id
    second_user_id = db.query(User).filter_by(email="second@example.com").one().id
    db.add_all(
        [
            Document(owner_id=first_user_id, filename="first.pdf", page_count=1),
            Document(owner_id=second_user_id, filename="second.pdf", page_count=1),
        ]
    )
    db.commit()

    assert test_client.get("/api/v1/documents").status_code == 401
    first_documents = test_client.get(
        "/api/v1/documents",
        headers={"Authorization": f"Bearer {first_token}"},
    )
    second_document_id = db.query(Document).filter_by(owner_id=second_user_id).one().id
    foreign_chat = test_client.post(
        "/api/v1/chat",
        headers={"Authorization": f"Bearer {first_token}"},
        json={"document_id": second_document_id, "message": "Question?"},
    )
    foreign_topics = test_client.get(
        f"/api/v1/documents/{second_document_id}/topics",
        headers={"Authorization": f"Bearer {first_token}"},
    )
    foreign_delete = test_client.delete(
        f"/api/v1/documents/{second_document_id}",
        headers={"Authorization": f"Bearer {first_token}"},
    )

    assert first_documents.status_code == 200
    assert [doc["filename"] for doc in first_documents.json()] == ["first.pdf"]
    assert foreign_chat.status_code == 404
    assert foreign_topics.status_code == 404
    assert foreign_delete.status_code == 404


def test_duplicate_registration_is_rejected(client) -> None:
    """An email address can only be registered once."""
    test_client, _ = client
    _register(test_client, "duplicate@example.com")
    response = test_client.post(
        "/api/v1/auth/register",
        json={"email": "duplicate@example.com", "password": "secure-password-123"},
    )
    assert response.status_code == 409


def test_existing_database_gets_nullable_document_owner_column() -> None:
    """Pre-auth document rows remain intact and inaccessible after migration."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    legacy = MetaData()
    documents = Table(
        "documents",
        legacy,
        Column("id", Integer, primary_key=True),
        Column("filename", String(255), nullable=False),
        Column("page_count", Integer, nullable=False),
        Column("created_at", DateTime),
    )
    legacy.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            documents.insert().values(id=4, filename="legacy.pdf", page_count=1)
        )

    initialize_database(engine)
    migrated_columns = {column["name"] for column in inspect(engine).get_columns("documents")}
    with engine.connect() as connection:
        row = connection.execute(documents.select()).one()
    engine.dispose()

    assert "owner_id" in migrated_columns
    assert row.filename == "legacy.pdf"
