# Legal Chatbot - Authentication Pydantic Schemas

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    """User roles"""
    ADMIN = "admin"
    SOLICITOR = "solicitor"
    PUBLIC = "public"


class OAuthProvider(str, Enum):
    """OAuth2 providers"""
    GOOGLE = "google"
    GITHUB = "github"
    MICROSOFT = "microsoft"


class UserBase(BaseModel):
    """Base user schema"""
    email: EmailStr
    username: Optional[str] = None
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None


class UserCreate(UserBase):
    """User creation schema"""
    password: str = Field(..., min_length=8, max_length=100)
    role: Optional[UserRole] = UserRole.PUBLIC


class UserUpdate(BaseModel):
    """User update schema"""
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8, max_length=100)
    is_active: Optional[bool] = None
    role: Optional[UserRole] = None


class UserResponse(BaseModel):
    """User response schema"""
    id: int
    email: str
    username: Optional[str] = None
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    role: UserRole
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    """Login request schema"""
    email: EmailStr
    password: str


class Token(BaseModel):
    """Token response schema"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class TokenRefresh(BaseModel):
    """Token refresh request schema"""
    refresh_token: str


class TokenData(BaseModel):
    """Token payload data"""
    user_id: int
    email: str
    role: UserRole


class OAuthLoginRequest(BaseModel):
    """OAuth login request"""
    provider: OAuthProvider
    code: str
    redirect_uri: Optional[str] = None


class OAuthAuthorizationURL(BaseModel):
    """OAuth authorization URL response"""
    authorization_url: str
    state: str


class PasswordChange(BaseModel):
    """Password change request"""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)


class UserListResponse(BaseModel):
    """User list response"""
    users: List[UserResponse]
    total: int
    page: int
    page_size: int


class UserStats(BaseModel):
    """User statistics"""
    total_users: int
    active_users: int
    verified_users: int
    users_by_role: dict[str, int]
