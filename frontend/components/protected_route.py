# Legal Chatbot - Protected Route Component
"""
Protected route decorator/component for Streamlit pages.
"""

import streamlit as st
from typing import List, Optional, Callable
from frontend.auth_ui import AuthUI


def protected_route(
    required_roles: Optional[List[str]] = None,
    redirect_to_login: bool = True
) -> Callable:
    """
    Decorator to protect Streamlit routes with authentication and role checking.
    
    Args:
        required_roles: List of roles allowed to access this route (None = any authenticated user)
        redirect_to_login: Whether to redirect to login page if not authenticated
    
    Usage:
        @protected_route(required_roles=["admin", "solicitor"])
        def admin_page():
            st.write("Admin content")
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            auth_ui = AuthUI()
            
            # Check authentication
            if not auth_ui.ensure_authenticated():
                if redirect_to_login:
                    auth_ui.render_authentication_page()
                    return
                else:
                    st.error("Authentication required")
                    return
            
            # Check role if required
            if required_roles:
                if not auth_ui.has_role(*required_roles):
                    st.error(f"Access denied. Required roles: {', '.join(required_roles)}")
                    st.info("You don't have permission to access this page.")
                    return
            
            # Execute the function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_role(*roles: str):
    """
    Helper function to check if user has required role.
    
    Args:
        roles: Required roles
    
    Returns:
        True if user has one of the required roles, False otherwise
    """
    auth_ui = AuthUI()
    
    if not auth_ui.ensure_authenticated():
        return False
    
    return auth_ui.has_role(*roles)

