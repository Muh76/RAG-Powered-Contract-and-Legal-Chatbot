# Legal Chatbot - Authentication Module

from app.auth.models import User, OAuthAccount, RefreshToken, UserRole
from app.auth.schemas import (
    UserCreate, UserUpdate, UserResponse,
    Token, TokenRefresh,
    OAuthProvider
)
from app.auth.oauth import OAuth2Provider
from app.auth.jwt import create_access_token, create_refresh_token, verify_token
from app.auth.dependencies import get_current_user, require_roles

__all__ = [
    "User",
    "UserRole",
    "OAuthAccount",
    "RefreshToken",
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    "Token",
    "TokenRefresh",
    "OAuthProvider",
    "OAuth2Provider",
    "create_access_token",
    "create_refresh_token",
    "verify_token",
    "get_current_user",
    "require_roles",
]

