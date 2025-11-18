# Legal Chatbot - Authentication Database Models

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, TypeDecorator
from sqlalchemy.dialects.postgresql import ENUM as PostgreSQL_ENUM
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import enum

Base = declarative_base()


class UserRole(str, enum.Enum):
    """User roles"""
    ADMIN = "admin"
    SOLICITOR = "solicitor"
    PUBLIC = "public"


class OAuthProvider(str, enum.Enum):
    """OAuth2 providers"""
    GOOGLE = "google"
    GITHUB = "github"
    MICROSOFT = "microsoft"


class EnumValueType(TypeDecorator):
    """TypeDecorator that stores enum values (not names) in the database"""
    impl = String
    cache_ok = True
    
    def __init__(self, enum_class, *args, **kwargs):
        self.enum_class = enum_class
        super().__init__(*args, **kwargs)
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, self.enum_class):
            return value.value
        return value
    
    def process_result_value(self, value, dialect):
        if value is None:
            return None
        try:
            return self.enum_class(value)
        except ValueError:
            return value


class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=True)
    hashed_password = Column(String(255), nullable=True)  # Nullable for OAuth-only users
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    role = Column(PostgreSQL_ENUM(UserRole, name='userrole', create_type=False, values_callable=lambda obj: [e.value for e in obj]), default=UserRole.PUBLIC, nullable=False)
    avatar_url = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    oauth_accounts = relationship("OAuthAccount", back_populates="user", cascade="all, delete-orphan")
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, role={self.role.value})>"


class OAuthAccount(Base):
    """OAuth2 account linking"""
    __tablename__ = "oauth_accounts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    provider = Column(PostgreSQL_ENUM(OAuthProvider, name='oauthprovider', create_type=False, values_callable=lambda obj: [e.value for e in obj]), nullable=False)
    provider_user_id = Column(String(255), nullable=False)
    provider_email = Column(String(255), nullable=True)
    access_token = Column(String(1000), nullable=True)  # Encrypted in production
    refresh_token = Column(String(1000), nullable=True)  # Encrypted in production
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="oauth_accounts")
    
    # Unique constraint: one user per provider
    __table_args__ = (
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"}
    )
    
    def __repr__(self):
        return f"<OAuthAccount(user_id={self.user_id}, provider={self.provider.value})>"


class RefreshToken(Base):
    """Refresh token model"""
    __tablename__ = "refresh_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    token = Column(String(500), unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    is_revoked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="refresh_tokens")
    
    def __repr__(self):
        return f"<RefreshToken(user_id={self.user_id}, expires_at={self.expires_at})>"

