# Legal Chatbot - Authentication API Routes

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional
import logging

from app.auth.service import AuthService
from app.auth.schemas import (
    UserCreate, UserUpdate, UserResponse, Token, TokenRefresh,
    OAuthProvider, OAuthLoginRequest, OAuthAuthorizationURL,
    PasswordChange, UserListResponse, UserStats, LoginRequest
)
from app.auth.dependencies import (
    get_current_user, get_current_active_user, require_admin,
    require_solicitor_or_admin
)
from app.auth.models import User, UserRole
from app.core.database import get_db
from app.core.errors import AuthenticationError, NotFoundError

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        user = AuthService.create_user(db, user_data)
        tokens = AuthService.create_tokens(user, db)
        return tokens
    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/login", response_model=Token)
async def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """Login with email and password"""
    try:
        user = AuthService.authenticate_user(db, login_data.email, login_data.password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        tokens = AuthService.create_tokens(user, db)
        return tokens
    except HTTPException:
        raise
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Login error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


@router.post("/refresh", response_model=Token)
async def refresh_token(token_data: TokenRefresh, db: Session = Depends(get_db)):
    """Refresh access token"""
    try:
        tokens = AuthService.refresh_access_token(db, token_data.refresh_token)
        return tokens
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/logout")
async def logout(
    token_data: TokenRefresh,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Logout and revoke refresh token"""
    AuthService.revoke_refresh_token(db, token_data.refresh_token, current_user.id)
    return {"message": "Successfully logged out"}


@router.post("/logout-all")
async def logout_all(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Logout from all devices"""
    AuthService.revoke_all_refresh_tokens(db, current_user.id)
    return {"message": "Successfully logged out from all devices"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(current_user: User = Depends(get_current_active_user)):
    """Get current user profile"""
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_current_user_profile(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update current user profile"""
    try:
        # Don't allow role changes via self-update
        if user_data.role:
            user_data.role = None
        
        updated_user = AuthService.update_user(db, current_user.id, user_data)
        return updated_user
    except (AuthenticationError, NotFoundError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Change user password"""
    if not current_user.hashed_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password authentication not available. Please use OAuth."
        )
    
    # Verify current password
    from app.auth.jwt import verify_password
    if not verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect current password"
        )
    
    # Update password
    user_data = UserUpdate(password=password_data.new_password)
    AuthService.update_user(db, current_user.id, user_data)
    
    return {"message": "Password changed successfully"}


# OAuth Routes
@router.get("/oauth/{provider}/authorize", response_model=OAuthAuthorizationURL)
async def oauth_authorize(
    provider: OAuthProvider,
    redirect_uri: str = Query(..., description="OAuth redirect URI"),
    state: Optional[str] = Query(None, description="State parameter for CSRF protection")
):
    """Get OAuth authorization URL"""
    try:
        auth_url, state_value = AuthService.get_oauth_authorization_url(
            provider, redirect_uri, state
        )
        return OAuthAuthorizationURL(authorization_url=auth_url, state=state_value)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/oauth/{provider}/callback", response_model=Token)
async def oauth_callback(
    provider: OAuthProvider,
    request: OAuthLoginRequest,
    db: Session = Depends(get_db)
):
    """OAuth callback endpoint"""
    try:
        tokens = AuthService.oauth_login(db, provider, request.code, request.redirect_uri or "")
        return tokens
    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


# Admin Routes
@router.get("/users", response_model=UserListResponse)
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    role: Optional[UserRole] = Query(None),
    is_active: Optional[bool] = Query(None),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """List all users (admin only)"""
    users, total = AuthService.list_users(db, skip, limit, role, is_active)
    return UserListResponse(
        users=[UserResponse.model_validate(user) for user in users],
        total=total,
        page=skip // limit + 1 if limit > 0 else 1,
        page_size=limit
    )


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get user by ID (admin only)"""
    user = AuthService.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Update user (admin only)"""
    try:
        updated_user = AuthService.update_user(db, user_id, user_data)
        return updated_user
    except (NotFoundError, AuthenticationError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Delete user (admin only - soft delete)"""
    try:
        AuthService.delete_user(db, user_id)
        return {"message": "User deleted successfully"}
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/stats", response_model=UserStats)
async def get_user_stats(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get user statistics (admin only)"""
    users, total = AuthService.list_users(db, skip=0, limit=10000)
    
    active_count = sum(1 for u in users if u.is_active)
    verified_count = sum(1 for u in users if u.is_verified)
    
    users_by_role = {}
    for role in UserRole:
        users_by_role[role.value] = sum(1 for u in users if u.role == role)
    
    return UserStats(
        total_users=total,
        active_users=active_count,
        verified_users=verified_count,
        users_by_role=users_by_role
    )

