# Legal Chatbot - Authentication Dependencies

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, List
from sqlalchemy.orm import Session

from app.auth.jwt import verify_token
from app.auth.schemas import UserRole, TokenData
from app.auth.models import User
from app.core.errors import AuthenticationError, AuthorizationError
from app.core.database import get_db

# HTTP Bearer token security
# auto_error=True means 403 is raised when no token provided (FastAPI default)
# This happens BEFORE get_current_user is called, so we can't customize the error message here
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user from JWT token"""
    token = credentials.credentials
    
    try:
        token_data = verify_token(token, token_type="access")
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    user = db.query(User).filter(User.id == token_data.user_id).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user (additional check)"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    return current_user


def require_roles(allowed_roles: List[UserRole]):
    """Dependency factory to require specific roles"""
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {[r.value for r in allowed_roles]}"
            )
        return current_user
    
    return role_checker


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require admin role"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def require_solicitor_or_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require solicitor or admin role"""
    if current_user.role not in [UserRole.SOLICITOR, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Solicitor or admin access required"
        )
    return current_user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user if authenticated, None otherwise"""
    if not credentials:
        return None
    
    try:
        token_data = verify_token(credentials.credentials, token_type="access")
        user = db.query(User).filter(User.id == token_data.user_id).first()
        return user if user and user.is_active else None
    except Exception:
        return None

