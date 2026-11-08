"""Authentication request and response schemas."""

from pydantic import BaseModel, EmailStr, Field, field_validator


class RegisterRequest(BaseModel):
    """New account registration details."""

    email: EmailStr
    password: str = Field(min_length=8, max_length=72)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        """Store email addresses in normalized lowercase form."""
        return value.strip().lower()

    @field_validator("password")
    @classmethod
    def bcrypt_password_limit(cls, value: str) -> str:
        """Reject passwords exceeding bcrypt's 72-byte input limit."""
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must be at most 72 UTF-8 bytes.")
        return value


class LoginRequest(BaseModel):
    """Existing account login details."""

    email: EmailStr
    password: str = Field(min_length=1, max_length=72)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        """Normalize email addresses before account lookup."""
        return value.strip().lower()

    @field_validator("password")
    @classmethod
    def bcrypt_password_limit(cls, value: str) -> str:
        """Reject passwords exceeding bcrypt's 72-byte input limit."""
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must be at most 72 UTF-8 bytes.")
        return value


class TokenResponse(BaseModel):
    """Bearer access token returned after authentication."""

    access_token: str
    token_type: str = "bearer"
