# Legal Chatbot - JWT Token Utilities

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.core.config import settings
from app.auth.schemas import TokenData, UserRole
from app.core.errors import AuthenticationError

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(
    user_id: int,
    email: str,
    role: UserRole,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token"""
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    expire = datetime.utcnow() + expires_delta
    
    payload = {
        "sub": str(user_id),  # Subject (user ID)
        "email": email,
        "role": role.value,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    }
    
    encoded_jwt = jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    
    return encoded_jwt


def create_refresh_token(
    user_id: int,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT refresh token"""
    if expires_delta is None:
        expires_delta = timedelta(days=7)  # Refresh tokens last 7 days
    
    expire = datetime.utcnow() + expires_delta
    
    payload = {
        "sub": str(user_id),
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    }
    
    encoded_jwt = jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> TokenData:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        # Check token type
        if payload.get("type") != token_type:
            raise AuthenticationError("Invalid token type")
        
        # Extract user data
        user_id = int(payload.get("sub"))
        
        if not user_id:
            raise AuthenticationError("Invalid token payload")
        
        # For refresh tokens, only user_id is needed
        if token_type == "refresh":
            # Return minimal TokenData for refresh tokens
            return TokenData(user_id=user_id, email="", role=UserRole.PUBLIC)
        
        # For access tokens, email and role are required
        email = payload.get("email")
        role_str = payload.get("role")
        
        if not email:
            raise AuthenticationError("Invalid token payload")
        
        # Validate role
        try:
            role = UserRole(role_str) if role_str else UserRole.PUBLIC
        except ValueError:
            role = UserRole.PUBLIC
        
        return TokenData(user_id=user_id, email=email, role=role)
    
    except JWTError as e:
        raise AuthenticationError(f"Invalid token: {str(e)}")
    except Exception as e:
        raise AuthenticationError(f"Token verification failed: {str(e)}")


def decode_token(token: str) -> dict:
    """Decode token without verification (for inspection)"""
    try:
        return jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"verify_signature": False}
        )
    except JWTError:
        return {}

