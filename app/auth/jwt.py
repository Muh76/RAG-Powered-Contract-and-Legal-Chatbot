# Legal Chatbot - JWT Token Utilities

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.core.config import settings
from app.auth.schemas import TokenData, UserRole
from app.core.errors import AuthenticationError

# Password hashing
# Use bcrypt with workaround for detection issues
import os
import logging

logger = logging.getLogger(__name__)

# Try to disable wrap bug detection
os.environ.setdefault("PASSLIB_WRAP_BUG_DETECTION", "0")

# Monkey-patch to avoid wrap bug detection issues
try:
    import passlib.handlers.bcrypt as bcrypt_module
    original_detect_wrap_bug = bcrypt_module.detect_wrap_bug
    def patched_detect_wrap_bug(ident):
        # Skip wrap bug detection to avoid 72-byte limit issues
        return False
    bcrypt_module.detect_wrap_bug = patched_detect_wrap_bug
except Exception:
    pass  # If patching fails, continue anyway

try:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
except Exception as e:
    logger.warning(f"Failed to initialize bcrypt context: {e}, trying alternative")
    try:
        # Try with explicit bcrypt backend and no auto-detection
        pwd_context = CryptContext(schemes=["bcrypt"], bcrypt__ident="2b", bcrypt__rounds=12)
    except Exception as e2:
        logger.error(f"Could not initialize password context: {e2}")
        pwd_context = None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    if not plain_password or not hashed_password:
        return False
    
    if pwd_context is None:
        # Fallback to direct bcrypt if pwd_context not available
        try:
            import bcrypt
            password_bytes = plain_password.encode('utf-8') if isinstance(plain_password, str) else plain_password
            if len(password_bytes) > 72:
                password_bytes = password_bytes[:72]
            hashed_bytes = hashed_password.encode('utf-8') if isinstance(hashed_password, str) else hashed_password
            return bcrypt.checkpw(password_bytes, hashed_bytes)
        except (ImportError, Exception):
            return False
    
    # Use pwd_context (passlib) - this was the original working version
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        # If passlib fails, try direct bcrypt as fallback
        try:
            import bcrypt
            password_bytes = plain_password.encode('utf-8') if isinstance(plain_password, str) else plain_password
            if len(password_bytes) > 72:
                password_bytes = password_bytes[:72]
            hashed_bytes = hashed_password.encode('utf-8') if isinstance(hashed_password, str) else hashed_password
            return bcrypt.checkpw(password_bytes, hashed_bytes)
        except (ImportError, Exception):
            return False


def get_password_hash(password: str) -> str:
    """Hash a password"""
    if not password:
        raise ValueError("Password cannot be empty")
    
    # Handle bcrypt's 72-byte limit by truncating before hashing
    if isinstance(password, str):
        password_bytes = password.encode('utf-8')
        if len(password_bytes) > 72:
            # Truncate to 72 bytes, then decode back to string
            password = password_bytes[:72].decode('utf-8', errors='ignore')
    
    if pwd_context is None:
        raise RuntimeError("Password context not initialized")
    
    try:
        return pwd_context.hash(password)
    except ValueError as e:
        # Handle bcrypt-specific errors
        error_str = str(e)
        if "72 bytes" in error_str:
            # Already truncated, try with bytes directly
            try:
                import bcrypt
                password_bytes = password.encode('utf-8')[:72]
                salt = bcrypt.gensalt()
                hashed = bcrypt.hashpw(password_bytes, salt)
                return hashed.decode('utf-8')
            except ImportError:
                raise ValueError("Password is too long (max 72 bytes) and bcrypt not available")
        raise
    except Exception as e:
        # Fallback to direct bcrypt if passlib fails
        try:
            import bcrypt
            password_bytes = password.encode('utf-8')[:72]
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password_bytes, salt)
            return hashed.decode('utf-8')
        except ImportError:
            raise RuntimeError(f"Password hashing failed: {e}")


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
        settings.get_jwt_secret(),
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
        settings.get_jwt_secret(),
        algorithm=settings.JWT_ALGORITHM
    )
    
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> TokenData:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(
            token,
            settings.get_jwt_secret(),
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
            settings.get_jwt_secret(),
            algorithms=[settings.JWT_ALGORITHM],
            options={"verify_signature": False}
        )
    except JWTError:
        return {}

