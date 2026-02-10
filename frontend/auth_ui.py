# Legal Chatbot - Authentication UI Components
"""
Streamlit authentication UI components for login, registration, and OAuth.
"""

import streamlit as st
import requests
import os
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import time


def _frontend_public_url() -> str:
    """Public URL of this frontend (for OAuth redirect). Set FRONTEND_PUBLIC_URL in production."""
    return os.environ.get("FRONTEND_PUBLIC_URL", "http://localhost:8501").rstrip("/")


class AuthUI:
    """Authentication UI components for Streamlit"""

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.session_state = st.session_state
        
        # Initialize session state
        if "access_token" not in self.session_state:
            self.session_state.access_token = None
        if "refresh_token" not in self.session_state:
            self.session_state.refresh_token = None
        if "token_expires_at" not in self.session_state:
            self.session_state.token_expires_at = None
        if "user" not in self.session_state:
            self.session_state.user = None
        if "is_authenticated" not in self.session_state:
            self.session_state.is_authenticated = False
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authorization headers for API requests"""
        if self.session_state.access_token:
            return {"Authorization": f"Bearer {self.session_state.access_token}"}
        return {}
    
    def is_token_expired(self) -> bool:
        """Check if access token is expired"""
        if not self.session_state.token_expires_at:
            return True
        
        # Add 60 second buffer before expiration
        buffer = timedelta(seconds=60)
        return datetime.now() >= self.session_state.token_expires_at - buffer
    
    def refresh_access_token(self) -> bool:
        """Refresh access token using refresh token"""
        if not self.session_state.refresh_token:
            return False
        
        try:
            response = requests.post(
                f"{self.api_base_url}/api/v1/auth/refresh",
                json={"refresh_token": self.session_state.refresh_token},
                timeout=10
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self._store_tokens(token_data)
                return True
            else:
                self.logout()
                return False
        except Exception as e:
            st.error(f"Token refresh failed: {str(e)}")
            self.logout()
            return False
    
    def _store_tokens(self, token_data: Dict[str, Any]):
        """Store tokens in session state"""
        self.session_state.access_token = token_data.get("access_token")
        self.session_state.refresh_token = token_data.get("refresh_token")
        
        # Calculate expiration time (default 30 minutes)
        expires_in = token_data.get("expires_in", 1800)  # 30 minutes in seconds
        self.session_state.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
    
    def _store_user(self, user_data: Dict[str, Any]):
        """Store user data in session state"""
        self.session_state.user = user_data
        self.session_state.is_authenticated = True
    
    def login(self, email: str, password: str) -> Tuple[bool, Optional[str]]:
        """
        Login with email and password.
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            response = requests.post(
                f"{self.api_base_url}/api/v1/auth/login",
                json={"email": email, "password": password},
                timeout=10
            )
            
            if response.status_code == 200:
                try:
                    token_data = response.json()
                    self._store_tokens(token_data)
                except (ValueError, requests.exceptions.JSONDecodeError) as e:
                    return False, f"Invalid response from server: {str(e)}"
                
                # Fetch user profile
                try:
                    user_response = requests.get(
                        f"{self.api_base_url}/api/v1/auth/me",
                        headers=self.get_auth_headers(),
                        timeout=10
                    )
                    
                    if user_response.status_code == 200:
                        user_data = user_response.json()
                        self._store_user(user_data)
                        return True, None
                    else:
                        # Tokens stored but couldn't fetch user - still success
                        return True, "Logged in but couldn't fetch user profile"
                except Exception as e:
                    # Tokens stored but couldn't fetch user - still success
                    return True, "Logged in but couldn't fetch user profile"
            else:
                # Handle error response
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", "Login failed")
                except (ValueError, requests.exceptions.JSONDecodeError):
                    # Response is not JSON, use status text
                    error_detail = f"Login failed: {response.status_code} {response.reason}"
                    if response.text:
                        error_detail += f" - {response.text[:200]}"
                
                return False, error_detail
                
        except requests.exceptions.Timeout:
            return False, "Connection timeout: Server took too long to respond"
        except requests.exceptions.ConnectionError:
            return False, "Connection error: Could not connect to server"
        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}"
        except Exception as e:
            return False, f"Login error: {str(e)}"
    
    def register(self, email: str, password: str, full_name: str = "", username: str = "") -> Tuple[bool, Optional[str]]:
        """
        Register a new user.
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            response = requests.post(
                f"{self.api_base_url}/api/v1/auth/register",
                json={
                    "email": email,
                    "password": password,
                    "full_name": full_name,
                    "username": username,
                    "role": "public"
                },
                timeout=10
            )
            
            if response.status_code == 201:
                token_data = response.json()
                self._store_tokens(token_data)
                
                # Fetch user profile
                user_response = requests.get(
                    f"{self.api_base_url}/api/v1/auth/me",
                    headers=self.get_auth_headers(),
                    timeout=10
                )
                
                if user_response.status_code == 200:
                    user_data = user_response.json()
                    self._store_user(user_data)
                    return True, None
                else:
                    return True, "Registered but couldn't fetch user profile"
            else:
                # Handle error response - check if it's JSON
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", f"Registration failed: {response.status_code}")
                except (ValueError, requests.exceptions.JSONDecodeError):
                    # Response is not JSON - likely HTML error page or empty response
                    if response.text:
                        error_detail = f"Registration failed: {response.status_code} - {response.text[:200]}"
                    else:
                        error_detail = f"Registration failed: {response.status_code} {response.reason}"
                
                return False, error_detail
                
        except requests.exceptions.Timeout:
            return False, "Connection timeout: Server took too long to respond"
        except requests.exceptions.ConnectionError:
            return False, f"Connection error: Could not connect to server. Make sure the backend is running at {self.api_base_url}"
        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}"
        except Exception as e:
            return False, f"Registration error: {str(e)}"
    
    def logout(self):
        """Logout and clear session state"""
        # Optionally revoke refresh token on server
        if self.session_state.refresh_token:
            try:
                requests.post(
                    f"{self.api_base_url}/api/v1/auth/logout",
                    json={"refresh_token": self.session_state.refresh_token},
                    headers=self.get_auth_headers(),
                    timeout=5
                )
            except:
                pass  # Ignore logout errors
        
        # Clear session state
        self.session_state.access_token = None
        self.session_state.refresh_token = None
        self.session_state.token_expires_at = None
        self.session_state.user = None
        self.session_state.is_authenticated = False
        self.session_state.messages = []  # Clear chat messages
    
    def get_user_role(self) -> Optional[str]:
        """Get current user role"""
        if self.session_state.user:
            return self.session_state.user.get("role", "public")
        return None
    
    def has_role(self, *roles: str) -> bool:
        """Check if user has any of the specified roles"""
        user_role = self.get_user_role()
        return user_role in roles if user_role else False
    
    def ensure_authenticated(self) -> bool:
        """Ensure user is authenticated, refresh token if needed"""
        if not self.session_state.is_authenticated:
            return False
        
        # Check if token needs refresh
        if self.is_token_expired():
            if not self.refresh_access_token():
                return False
        
        return True
    
    def render_login_form(self) -> bool:
        """
        Render login form.
        
        Returns:
            True if login successful, False otherwise
        """
        st.subheader("Sign in")
        
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="your.email@example.com")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if not email or not password:
                    st.error("Please enter both email and password")
                    return False
                
                with st.spinner("Logging in..."):
                    success, error = self.login(email, password)
                
                if success:
                    st.success("Login successful! Redirecting...")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"Login failed: {error}")
                
                return success
        
        return False
    
    def render_register_form(self) -> bool:
        """
        Render registration form.
        
        Returns:
            True if registration successful, False otherwise
        """
        st.subheader("Create account")
        
        with st.form("register_form"):
            email = st.text_input("Email", placeholder="your.email@example.com")
            password = st.text_input("Password", type="password", placeholder="At least 8 characters")
            password_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
            full_name = st.text_input("Full Name (Optional)", placeholder="John Doe")
            username = st.text_input("Username (Optional)", placeholder="johndoe")
            
            submit = st.form_submit_button("Register", use_container_width=True)
            
            if submit:
                if not email or not password:
                    st.error("Email and password are required")
                    return False
                
                if password != password_confirm:
                    st.error("Passwords do not match")
                    return False
                
                if len(password) < 8:
                    st.error("Password must be at least 8 characters")
                    return False
                
                with st.spinner("Registering..."):
                    success, error = self.register(email, password, full_name, username)
                
                if success:
                    st.success("Registration successful! Redirecting...")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"Registration failed: {error}")
                
                return success
        
        return False
    
    def render_oauth_buttons(self):
        """Render OAuth login buttons"""
        st.markdown("---")
        st.caption("Or continue with")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Google", use_container_width=True, key="oauth_google"):
                self._initiate_oauth("google")
        
        with col2:
            if st.button("GitHub", use_container_width=True, key="oauth_github"):
                self._initiate_oauth("github")
        
        with col3:
            if st.button("Microsoft", use_container_width=True, key="oauth_microsoft"):
                self._initiate_oauth("microsoft")
    
    def _initiate_oauth(self, provider: str):
        """Initiate OAuth flow"""
        try:
            # Get OAuth authorization URL
            redirect_uri = f"{_frontend_public_url()}/auth/callback"
            response = requests.get(
                f"{self.api_base_url}/api/v1/auth/oauth/{provider}/authorize",
                params={"redirect_uri": redirect_uri},
                timeout=10,
                allow_redirects=False  # Don't follow redirects
            )
            
            if response.status_code == 302:
                # Extract redirect URL from Location header
                auth_url = response.headers.get("Location")
                if auth_url:
                    st.info(f"Redirecting to {provider}...")
                    st.markdown(f"[Click here to continue with {provider}]({auth_url})")
                    
                    # Store OAuth state in session
                    self.session_state.oauth_provider = provider
                    self.session_state.oauth_redirect_url = auth_url
                else:
                    st.error(f"OAuth {provider} authorization URL not found")
            elif response.status_code == 200:
                # Try to get JSON response
                try:
                    data = response.json()
                    auth_url = data.get("authorization_url")
                    if auth_url:
                        st.info(f"Redirecting to {provider}...")
                        st.markdown(f"[Click here to continue with {provider}]({auth_url})")
                        self.session_state.oauth_provider = provider
                        self.session_state.oauth_redirect_url = auth_url
                    else:
                        st.error(f"OAuth {provider} authorization URL not found in response")
                except:
                    st.error(f"OAuth {provider} response format error")
            else:
                error_detail = response.json().get("detail", "Unknown error") if response.text else "OAuth not available"
                st.error(f"OAuth {provider} error: {error_detail}")
        except Exception as e:
            st.error(f"OAuth error: {str(e)}")
    
    def render_user_profile(self):
        """Render user profile display — visual clarity only, no logic changes"""
        if not self.session_state.user:
            return
        
        user = self.session_state.user
        role = user.get("role", "public")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Signed in**", help="Current user and access level")
        
        # User name (primary)
        name = user.get("full_name") or user.get("email", "User")
        st.sidebar.markdown(f'<span class="sidebar-user-name">{name}</span>', unsafe_allow_html=True)
        
        # Email (secondary)
        if user.get("email") and user.get("full_name"):
            st.sidebar.markdown(f'<span class="sidebar-user-email">{user["email"]}</span>', unsafe_allow_html=True)
        
        # Role badge: Public / Solicitor / Admin
        role_class = f"sidebar-role-{role}"
        role_label = role.capitalize()
        st.sidebar.markdown(f'<span class="sidebar-role-badge {role_class}">{role_label}</span>', unsafe_allow_html=True)
        st.sidebar.caption("Access level")
        st.sidebar.markdown("")  # spacing
        
        # Logout button
        if st.sidebar.button("Sign out", use_container_width=True):
            self.logout()
            st.rerun()
    
    def render_authentication_page(self):
        """Render main authentication page with login/register tabs"""
        st.title("Legal Chatbot")
        st.caption("Sign in to access the legal assistant.")
        st.caption("Sign in or create an account to get started.")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            self.render_login_form()
            self.render_oauth_buttons()
        
        with tab2:
            self.render_register_form()
            st.caption("Already have an account? Switch to the Login tab.")
    
    def handle_oauth_callback(self, code: str, state: Optional[str] = None, provider: Optional[str] = None):
        """Handle OAuth callback"""
        if not provider:
            # Try to detect provider from query params or session
            provider = self.session_state.get("oauth_provider")
            if not provider:
                # Default to google if not specified
                provider = "google"
        
        try:
            # OAuth callback endpoint is POST
            # Provider is in path, but OAuthLoginRequest schema also requires provider in body
            redirect_uri = f"{_frontend_public_url()}/auth/callback"
            response = requests.post(
                f"{self.api_base_url}/api/v1/auth/oauth/{provider}/callback",
                json={
                    "provider": provider,
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
                timeout=10,
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self._store_tokens(token_data)
                
                # Fetch user profile
                user_response = requests.get(
                    f"{self.api_base_url}/api/v1/auth/me",
                    headers=self.get_auth_headers(),
                    timeout=10
                )
                
                if user_response.status_code == 200:
                    user_data = user_response.json()
                    self._store_user(user_data)
                    return True
            else:
                error_detail = response.json().get("detail", "Unknown error") if response.text else "OAuth callback failed"
                st.error(f"OAuth callback failed: {error_detail}")
                return False
        except Exception as e:
            st.error(f"OAuth callback error: {str(e)}")
            return False

