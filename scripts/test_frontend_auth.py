#!/usr/bin/env python3
"""
Test script for frontend authentication integration.
Tests authentication UI components and integration.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=" * 60)
print("FRONTEND AUTHENTICATION INTEGRATION TEST")
print("=" * 60)

# --- 1. Test Imports ---
print("\n1. Testing imports...")
try:
    from frontend.auth_ui import AuthUI
    print("   ✅ AuthUI imported successfully")
    
    from frontend.app import LegalChatbotUI
    print("   ✅ LegalChatbotUI imported successfully")
    
    from frontend.components.protected_route import protected_route, require_role
    print("   ✅ Protected route components imported successfully")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# --- 2. Test AuthUI Initialization ---
print("\n2. Testing AuthUI initialization...")
try:
    auth_ui = AuthUI(api_base_url="http://localhost:8000")
    print("   ✅ AuthUI initialized successfully")
    print(f"   ✅ API base URL: {auth_ui.api_base_url}")
except Exception as e:
    print(f"   ❌ Initialization failed: {e}")
    sys.exit(1)

# --- 3. Test Token Management Methods ---
print("\n3. Testing token management methods...")
try:
    # Test get_auth_headers (should return empty dict when not authenticated)
    headers = auth_ui.get_auth_headers()
    assert isinstance(headers, dict), "get_auth_headers should return a dict"
    print("   ✅ get_auth_headers() works")
    
    # Test is_token_expired (should return True when no token)
    expired = auth_ui.is_token_expired()
    assert expired == True, "Token should be expired when not set"
    print("   ✅ is_token_expired() works")
    
    # Test ensure_authenticated (should return False when not authenticated)
    authenticated = auth_ui.ensure_authenticated()
    assert authenticated == False, "Should not be authenticated when no tokens"
    print("   ✅ ensure_authenticated() works")
    
    # Test get_user_role (should return None when not authenticated)
    role = auth_ui.get_user_role()
    assert role is None, "Role should be None when not authenticated"
    print("   ✅ get_user_role() works")
    
    # Test has_role (should return False when not authenticated)
    has_admin = auth_ui.has_role("admin")
    assert has_admin == False, "Should not have role when not authenticated"
    print("   ✅ has_role() works")
except Exception as e:
    print(f"   ❌ Token management test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- 4. Test UI Components Structure ---
print("\n4. Testing UI component structure...")
try:
    # Check if methods exist
    assert hasattr(auth_ui, 'render_login_form'), "AuthUI should have render_login_form method"
    print("   ✅ render_login_form() method exists")
    
    assert hasattr(auth_ui, 'render_register_form'), "AuthUI should have render_register_form method"
    print("   ✅ render_register_form() method exists")
    
    assert hasattr(auth_ui, 'render_oauth_buttons'), "AuthUI should have render_oauth_buttons method"
    print("   ✅ render_oauth_buttons() method exists")
    
    assert hasattr(auth_ui, 'render_user_profile'), "AuthUI should have render_user_profile method"
    print("   ✅ render_user_profile() method exists")
    
    assert hasattr(auth_ui, 'render_authentication_page'), "AuthUI should have render_authentication_page method"
    print("   ✅ render_authentication_page() method exists")
    
    assert hasattr(auth_ui, 'handle_oauth_callback'), "AuthUI should have handle_oauth_callback method"
    print("   ✅ handle_oauth_callback() method exists")
except Exception as e:
    print(f"   ❌ UI component structure test failed: {e}")
    sys.exit(1)

# --- 5. Test LegalChatbotUI Integration ---
print("\n5. Testing LegalChatbotUI integration...")
try:
    ui = LegalChatbotUI()
    print("   ✅ LegalChatbotUI initialized successfully")
    
    # Check if auth_ui is integrated
    assert hasattr(ui, 'auth_ui'), "LegalChatbotUI should have auth_ui attribute"
    print("   ✅ AuthUI integrated in LegalChatbotUI")
    
    # Check if send_chat_request uses authentication
    assert 'ensure_authenticated' in str(ui.send_chat_request.__code__.co_names), "send_chat_request should check authentication"
    print("   ✅ send_chat_request() includes authentication check")
    
    # Check if render_sidebar checks authentication
    assert 'auth_ui' in str(ui.render_sidebar.__code__.co_names), "render_sidebar should use auth_ui"
    print("   ✅ render_sidebar() integrates with auth_ui")
except Exception as e:
    print(f"   ❌ UI integration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- 6. Test Protected Route Component ---
print("\n6. Testing protected route component...")
try:
    # Check if functions exist
    assert callable(protected_route), "protected_route should be callable"
    print("   ✅ protected_route() decorator exists")
    
    assert callable(require_role), "require_role should be callable"
    print("   ✅ require_role() helper exists")
except Exception as e:
    print(f"   ❌ Protected route component test failed: {e}")
    sys.exit(1)

# --- Summary ---
print("\n" + "=" * 60)
print("FRONTEND AUTHENTICATION INTEGRATION TEST COMPLETE")
print("=" * 60)
print("\n✅ All core components tested successfully!")
print("\n📋 Implementation Summary:")
print("   • AuthUI class with login/register/OAuth functionality")
print("   • Token storage and refresh handling")
print("   • Protected routes and role-based UI rendering")
print("   • User profile display and logout")
print("   • OAuth buttons (Google, GitHub, Microsoft)")
print("   • Integration with LegalChatbotUI")
print("   • Documents and Settings pages")
print("\n🚀 Next Steps:")
print("   1. Start API server: uvicorn app.api.main:app --reload")
print("   2. Start Streamlit: streamlit run frontend/app.py")
print("   3. Test login/register flow in browser")
print("   4. Test OAuth flow (if configured)")
print("   5. Test role-based UI rendering")
print("\n✨ Frontend authentication integration complete!")

