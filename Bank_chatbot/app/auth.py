import hashlib
import secrets
from typing import Dict

from fastapi import HTTPException, status

from app.database import get_user_by_username

ACTIVE_TOKENS: Dict[str, str] = {}


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def generate_token(user_id: str) -> str:
    token = secrets.token_urlsafe(24)
    ACTIVE_TOKENS[token] = user_id
    return token


def validate_token(token: str) -> str:
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")

    raw_token = token.replace("Bearer ", "").strip()
    if raw_token not in ACTIVE_TOKENS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    return ACTIVE_TOKENS[raw_token]


def login_user(username: str, password: str):
    user = get_user_by_username(username)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

    if hash_password(password) != user.get("password_hash"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

    token = generate_token(str(user["user_id"]))
    return {"message": "Login successful", "token": token}
