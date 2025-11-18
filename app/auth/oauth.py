# Legal Chatbot - OAuth2 Provider Implementation

import secrets
import httpx
from typing import Optional, Dict, Any
from urllib.parse import urlencode, parse_qs, urlparse
from app.core.config import settings
from app.auth.schemas import OAuthProvider
from app.core.errors import AuthenticationError


class OAuth2Provider:
    """OAuth2 provider base class"""
    
    def __init__(self, provider: OAuthProvider):
        self.provider = provider
        self.config = self._get_config()
    
    def _get_config(self) -> Dict[str, str]:
        """Get provider configuration from settings"""
        provider_name = self.provider.value.upper()
        
        config = {
            "client_id": getattr(settings, f"OAUTH_{provider_name}_CLIENT_ID", ""),
            "client_secret": getattr(settings, f"OAUTH_{provider_name}_CLIENT_SECRET", ""),
        }
        
        if not config["client_id"] or not config["client_secret"]:
            raise ValueError(f"OAuth {provider_name} credentials not configured")
        
        return config
    
    def get_authorization_url(self, redirect_uri: str, state: Optional[str] = None) -> tuple[str, str]:
        """Get OAuth authorization URL"""
        if state is None:
            state = secrets.token_urlsafe(32)
        
        params = {
            "client_id": self.config["client_id"],
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": self._get_scopes(),
            "state": state,
        }
        
        url = f"{self._get_auth_url()}?{urlencode(params)}"
        return url, state
    
    def exchange_code_for_token(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange authorization code for access token"""
        token_url = self._get_token_url()
        
        data = {
            "client_id": self.config["client_id"],
            "client_secret": self.config["client_secret"],
            "code": code,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
        
        try:
            response = httpx.post(token_url, data=data, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise AuthenticationError(f"OAuth token exchange failed: {str(e)}")
    
    def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information from provider"""
        user_info_url = self._get_user_info_url()
        
        headers = {"Authorization": f"Bearer {access_token}"}
        
        try:
            response = httpx.get(user_info_url, headers=headers, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise AuthenticationError(f"Failed to fetch user info: {str(e)}")
    
    def _get_auth_url(self) -> str:
        """Get authorization URL (implemented by subclasses)"""
        raise NotImplementedError
    
    def _get_token_url(self) -> str:
        """Get token URL (implemented by subclasses)"""
        raise NotImplementedError
    
    def _get_user_info_url(self) -> str:
        """Get user info URL (implemented by subclasses)"""
        raise NotImplementedError
    
    def _get_scopes(self) -> str:
        """Get OAuth scopes (implemented by subclasses)"""
        raise NotImplementedError


class GoogleOAuthProvider(OAuth2Provider):
    """Google OAuth2 provider"""
    
    def __init__(self):
        super().__init__(OAuthProvider.GOOGLE)
    
    def _get_auth_url(self) -> str:
        return "https://accounts.google.com/o/oauth2/v2/auth"
    
    def _get_token_url(self) -> str:
        return "https://oauth2.googleapis.com/token"
    
    def _get_user_info_url(self) -> str:
        return "https://www.googleapis.com/oauth2/v2/userinfo"
    
    def _get_scopes(self) -> str:
        return "openid email profile"
    
    def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user info from Google"""
        user_data = super().get_user_info(access_token)
        
        return {
            "provider_user_id": user_data.get("id"),
            "email": user_data.get("email"),
            "full_name": user_data.get("name"),
            "avatar_url": user_data.get("picture"),
            "is_verified": user_data.get("verified_email", False),
        }


class GitHubOAuthProvider(OAuth2Provider):
    """GitHub OAuth2 provider"""
    
    def __init__(self):
        super().__init__(OAuthProvider.GITHUB)
    
    def _get_auth_url(self) -> str:
        return "https://github.com/login/oauth/authorize"
    
    def _get_token_url(self) -> str:
        return "https://github.com/login/oauth/access_token"
    
    def _get_user_info_url(self) -> str:
        return "https://api.github.com/user"
    
    def _get_scopes(self) -> str:
        return "user:email"
    
    def exchange_code_for_token(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange code for token (GitHub returns different format)"""
        token_url = self._get_token_url()
        
        data = {
            "client_id": self.config["client_id"],
            "client_secret": self.config["client_secret"],
            "code": code,
            "redirect_uri": redirect_uri,
        }
        
        headers = {"Accept": "application/json"}
        
        try:
            response = httpx.post(token_url, data=data, headers=headers, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise AuthenticationError(f"OAuth token exchange failed: {str(e)}")
    
    def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user info from GitHub"""
        user_data = super().get_user_info(access_token)
        
        # Get email (may require additional request)
        email = user_data.get("email")
        if not email:
            try:
                emails_response = httpx.get(
                    "https://api.github.com/user/emails",
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=10.0
                )
                emails_response.raise_for_status()
                emails = emails_response.json()
                primary_email = next((e for e in emails if e.get("primary")), emails[0] if emails else None)
                email = primary_email.get("email") if primary_email else None
            except Exception:
                pass
        
        return {
            "provider_user_id": str(user_data.get("id")),
            "email": email,
            "full_name": user_data.get("name"),
            "avatar_url": user_data.get("avatar_url"),
            "is_verified": user_data.get("email") is not None,
        }


class MicrosoftOAuthProvider(OAuth2Provider):
    """Microsoft OAuth2 provider"""
    
    def __init__(self):
        super().__init__(OAuthProvider.MICROSOFT)
        self.tenant_id = getattr(settings, "OAUTH_MICROSOFT_TENANT_ID", "common")
    
    def _get_auth_url(self) -> str:
        return f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/authorize"
    
    def _get_token_url(self) -> str:
        return f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
    
    def _get_user_info_url(self) -> str:
        return "https://graph.microsoft.com/v1.0/me"
    
    def _get_scopes(self) -> str:
        return "openid email profile User.Read"
    
    def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user info from Microsoft Graph"""
        user_data = super().get_user_info(access_token)
        
        return {
            "provider_user_id": user_data.get("id"),
            "email": user_data.get("mail") or user_data.get("userPrincipalName"),
            "full_name": user_data.get("displayName"),
            "avatar_url": None,  # Microsoft Graph requires separate call for photo
            "is_verified": True,  # Microsoft accounts are verified
        }


def get_oauth_provider(provider: OAuthProvider) -> OAuth2Provider:
    """Get OAuth provider instance"""
    providers = {
        OAuthProvider.GOOGLE: GoogleOAuthProvider,
        OAuthProvider.GITHUB: GitHubOAuthProvider,
        OAuthProvider.MICROSOFT: MicrosoftOAuthProvider,
    }
    
    provider_class = providers.get(provider)
    if not provider_class:
        raise ValueError(f"Unsupported OAuth provider: {provider}")
    
    return provider_class()

