# Legal Chatbot - Authentication Service

from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.auth.models import User, OAuthAccount, RefreshToken, UserRole, OAuthProvider
from app.auth.schemas import UserCreate, UserUpdate, OAuthProvider as OAuthProviderEnum
from app.auth.jwt import (
    create_access_token, create_refresh_token, verify_token,
    get_password_hash, verify_password
)
from app.auth.oauth import get_oauth_provider
from app.core.errors import AuthenticationError, NotFoundError
from app.core.config import settings
from app.core.database import SessionLocal


class AuthService:
    """Authentication service"""
    
    @staticmethod
    def create_user(db: Session, user_data: UserCreate, role: Optional[UserRole] = None) -> User:
        """Create a new user"""
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise AuthenticationError("User with this email already exists")
        
        if user_data.username:
            existing_username = db.query(User).filter(User.username == user_data.username).first()
            if existing_username:
                raise AuthenticationError("Username already taken")
        
        # Hash password
        hashed_password = get_password_hash(user_data.password) if user_data.password else None
        
        # Create user
        user = User(
            email=user_data.email,
            username=user_data.username,
            hashed_password=hashed_password,
            full_name=user_data.full_name,
            role=role or user_data.role or UserRole.PUBLIC,
            avatar_url=user_data.avatar_url,
            is_active=True,
            is_verified=False,  # Email verification required
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        return user
    
    @staticmethod
    def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        try:
            user = db.query(User).filter(User.email == email).first()
            
            if not user:
                return None
            
            if not user.hashed_password:
                raise AuthenticationError("Password authentication not available. Please use OAuth.")
            
            # Verify password with error handling
            try:
                if not verify_password(password, user.hashed_password):
                    return None
            except Exception as e:
                # Log password verification errors but don't expose details
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Password verification error for user {email}: {e}")
                return None
            
            if not user.is_active:
                raise AuthenticationError("User account is inactive")
            
            # Update last login
            user.last_login = datetime.utcnow()
            db.commit()
            
            return user
        except AuthenticationError:
            raise
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Authentication error: {e}", exc_info=True)
            return None
    
    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
        """Get user by ID"""
        return db.query(User).filter(User.id == user_id).first()
    
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email"""
        return db.query(User).filter(User.email == email).first()
    
    @staticmethod
    def update_user(db: Session, user_id: int, user_data: UserUpdate) -> User:
        """Update user information"""
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise NotFoundError("User not found")
        
        # Update fields
        if user_data.email is not None:
            # Check if email is already taken by another user
            existing = db.query(User).filter(
                and_(User.email == user_data.email, User.id != user_id)
            ).first()
            if existing:
                raise AuthenticationError("Email already taken")
            user.email = user_data.email
        
        if user_data.username is not None:
            existing = db.query(User).filter(
                and_(User.username == user_data.username, User.id != user_id)
            ).first()
            if existing:
                raise AuthenticationError("Username already taken")
            user.username = user_data.username
        
        if user_data.full_name is not None:
            user.full_name = user_data.full_name
        
        if user_data.avatar_url is not None:
            user.avatar_url = user_data.avatar_url
        
        if user_data.password is not None:
            user.hashed_password = get_password_hash(user_data.password)
        
        if user_data.is_active is not None:
            user.is_active = user_data.is_active
        
        if user_data.role is not None:
            user.role = user_data.role
        
        user.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(user)
        
        return user
    
    @staticmethod
    def create_tokens(user: User, db: Optional[Session] = None) -> dict:
        """Create access and refresh tokens for user"""
        access_token = create_access_token(
            user_id=user.id,
            email=user.email,
            role=user.role
        )
        
        refresh_token_str = create_refresh_token(user_id=user.id)
        
        # Store refresh token in database
        expires_at = datetime.utcnow() + timedelta(days=7)
        refresh_token = RefreshToken(
            user_id=user.id,
            token=refresh_token_str,
            expires_at=expires_at,
            is_revoked=False
        )
        
        # Use provided db session or create new one
        if db is None:
            from app.core.database import SessionLocal
            db = SessionLocal()
            try:
                db.add(refresh_token)
                db.commit()
            finally:
                db.close()
        else:
            db.add(refresh_token)
            db.commit()
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token_str,
            "token_type": "bearer",
            "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    
    @staticmethod
    def refresh_access_token(db: Session, refresh_token_str: str) -> dict:
        """Refresh access token using refresh token"""
        # Verify refresh token
        try:
            token_data = verify_token(refresh_token_str, token_type="refresh")
        except AuthenticationError:
            raise AuthenticationError("Invalid refresh token")
        
        # Check if token exists in database and is not revoked
        refresh_token = db.query(RefreshToken).filter(
            and_(
                RefreshToken.token == refresh_token_str,
                RefreshToken.user_id == token_data.user_id,
                RefreshToken.is_revoked == False,
                RefreshToken.expires_at > datetime.utcnow()
            )
        ).first()
        
        if not refresh_token:
            raise AuthenticationError("Invalid or expired refresh token")
        
        # Get user
        user = db.query(User).filter(User.id == token_data.user_id).first()
        if not user or not user.is_active:
            raise AuthenticationError("User not found or inactive")
        
        # Update refresh token last used
        refresh_token.last_used_at = datetime.utcnow()
        db.commit()
        
        # Create new access token
        access_token = create_access_token(
            user_id=user.id,
            email=user.email,
            role=user.role
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token_str,  # Reuse same refresh token
            "token_type": "bearer",
            "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    
    @staticmethod
    def revoke_refresh_token(db: Session, refresh_token_str: str, user_id: int):
        """Revoke a refresh token"""
        refresh_token = db.query(RefreshToken).filter(
            and_(
                RefreshToken.token == refresh_token_str,
                RefreshToken.user_id == user_id
            )
        ).first()
        
        if refresh_token:
            refresh_token.is_revoked = True
            db.commit()
    
    @staticmethod
    def revoke_all_refresh_tokens(db: Session, user_id: int):
        """Revoke all refresh tokens for a user"""
        db.query(RefreshToken).filter(RefreshToken.user_id == user_id).update(
            {"is_revoked": True}
        )
        db.commit()
    
    @staticmethod
    def oauth_login(
        db: Session,
        provider: OAuthProviderEnum,
        code: str,
        redirect_uri: str
    ) -> dict:
        """OAuth login flow"""
        # Get OAuth provider
        oauth_provider = get_oauth_provider(provider)
        
        # Exchange code for token
        token_response = oauth_provider.exchange_code_for_token(code, redirect_uri)
        access_token = token_response.get("access_token") or token_response.get("access_token")
        
        if not access_token:
            raise AuthenticationError("Failed to get access token from OAuth provider")
        
        # Get user info
        user_info = oauth_provider.get_user_info(access_token)
        
        # Check if OAuth account exists
        oauth_account = db.query(OAuthAccount).filter(
            and_(
                OAuthAccount.provider == OAuthProvider(provider.value),
                OAuthAccount.provider_user_id == str(user_info["provider_user_id"])
            )
        ).first()
        
        if oauth_account:
            # Existing user - update tokens and login
            user = oauth_account.user
            oauth_account.access_token = access_token  # In production, encrypt this
            oauth_account.refresh_token = token_response.get("refresh_token")
            oauth_account.updated_at = datetime.utcnow()
            user.last_login = datetime.utcnow()
            db.commit()
        else:
            # New user - check if email exists
            user = db.query(User).filter(User.email == user_info["email"]).first()
            
            if user:
                # Link OAuth account to existing user
                oauth_account = OAuthAccount(
                    user_id=user.id,
                    provider=OAuthProvider(provider.value),
                    provider_user_id=str(user_info["provider_user_id"]),
                    provider_email=user_info["email"],
                    access_token=access_token,
                    refresh_token=token_response.get("refresh_token"),
                )
                db.add(oauth_account)
                user.last_login = datetime.utcnow()
            else:
                # Create new user
                user = User(
                    email=user_info["email"],
                    username=None,
                    hashed_password=None,  # OAuth-only user
                    full_name=user_info.get("full_name"),
                    avatar_url=user_info.get("avatar_url"),
                    role=UserRole.PUBLIC,
                    is_active=True,
                    is_verified=user_info.get("is_verified", True),
                )
                db.add(user)
                db.flush()
                
                oauth_account = OAuthAccount(
                    user_id=user.id,
                    provider=OAuthProvider(provider.value),
                    provider_user_id=str(user_info["provider_user_id"]),
                    provider_email=user_info["email"],
                    access_token=access_token,
                    refresh_token=token_response.get("refresh_token"),
                )
                db.add(oauth_account)
            
            db.commit()
            db.refresh(user)
        
        # Create tokens
        return AuthService.create_tokens(user, db)
    
    @staticmethod
    def get_oauth_authorization_url(
        provider: OAuthProviderEnum,
        redirect_uri: str,
        state: Optional[str] = None
    ) -> tuple[str, str]:
        """Get OAuth authorization URL"""
        oauth_provider = get_oauth_provider(provider)
        return oauth_provider.get_authorization_url(redirect_uri, state)
    
    @staticmethod
    def list_users(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        role: Optional[UserRole] = None,
        is_active: Optional[bool] = None
    ) -> tuple[list[User], int]:
        """List users with filters"""
        query = db.query(User)
        
        if role:
            query = query.filter(User.role == role)
        
        if is_active is not None:
            query = query.filter(User.is_active == is_active)
        
        total = query.count()
        users = query.offset(skip).limit(limit).all()
        
        return users, total
    
    @staticmethod
    def delete_user(db: Session, user_id: int):
        """Delete user (soft delete by setting is_active=False)"""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise NotFoundError("User not found")
        
        user.is_active = False
        db.commit()


# Import SessionLocal here to avoid circular import
from app.core.database import SessionLocal

