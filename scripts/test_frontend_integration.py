#!/usr/bin/env python3
"""
Integration test for frontend authentication.
Tests the full authentication flow with the API server.
"""

import requests
import time
import sys
from typing import Dict, Any, Optional

API_BASE_URL = "http://localhost:8000/api/v1"

def test_health():
    """Test API health endpoint"""
    print("1. Testing API health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ API server is running")
            return True
        else:
            print(f"   ❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Cannot connect to API server: {e}")
        return False

def test_register():
    """Test user registration"""
    print("\n2. Testing user registration...")
    try:
        test_email = f"test_user_{int(time.time())}@example.com"
        response = requests.post(
            f"{API_BASE_URL}/auth/register",
            json={
                "email": test_email,
                "password": "testpassword123",
                "full_name": "Test User",
                "username": f"testuser_{int(time.time())}",
                "role": "public"
            },
            timeout=10
        )
        
        if response.status_code == 201:
            data = response.json()
            if "access_token" in data and "refresh_token" in data:
                print(f"   ✅ Registration successful for {test_email}")
                return data, test_email
            else:
                print(f"   ❌ Registration response missing tokens")
                return None, None
        else:
            error = response.json().get("detail", "Unknown error")
            print(f"   ❌ Registration failed: {error}")
            return None, None
    except Exception as e:
        print(f"   ❌ Registration error: {e}")
        return None, None

def test_login(email: str, password: str):
    """Test user login"""
    print("\n3. Testing user login...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/login",
            json={
                "email": email,
                "password": password
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if "access_token" in data and "refresh_token" in data:
                print(f"   ✅ Login successful")
                return data
            else:
                print(f"   ❌ Login response missing tokens")
                return None
        else:
            error = response.json().get("detail", "Unknown error")
            print(f"   ❌ Login failed: {error}")
            return None
    except Exception as e:
        print(f"   ❌ Login error: {e}")
        return None

def test_get_profile(access_token: str):
    """Test getting user profile"""
    print("\n4. Testing get user profile...")
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(
            f"{API_BASE_URL}/auth/me",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            user_data = response.json()
            print(f"   ✅ Profile retrieved successfully")
            print(f"      Email: {user_data.get('email')}")
            print(f"      Role: {user_data.get('role')}")
            print(f"      Full Name: {user_data.get('full_name')}")
            return user_data
        else:
            error = response.json().get("detail", "Unknown error")
            print(f"   ❌ Get profile failed: {error}")
            return None
    except Exception as e:
        print(f"   ❌ Get profile error: {e}")
        return None

def test_refresh_token(refresh_token: str):
    """Test token refresh"""
    print("\n5. Testing token refresh...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/refresh",
            json={"refresh_token": refresh_token},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if "access_token" in data:
                print(f"   ✅ Token refresh successful")
                return data.get("access_token")
            else:
                print(f"   ❌ Refresh response missing access_token")
                return None
        else:
            error = response.json().get("detail", "Unknown error")
            print(f"   ❌ Token refresh failed: {error}")
            return None
    except Exception as e:
        print(f"   ❌ Token refresh error: {e}")
        return None

def test_chat_endpoint(access_token: str):
    """Test chat endpoint with authentication"""
    print("\n6. Testing chat endpoint (requires auth)...")
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.post(
            f"{API_BASE_URL}/chat",
            headers=headers,
            json={
                "query": "What is contract law?",
                "top_k": 3
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Chat endpoint accessible")
            print(f"      Answer length: {len(data.get('answer', ''))} characters")
            return True
        elif response.status_code == 401:
            print(f"   ❌ Chat endpoint requires authentication (401)")
            return False
        else:
            error = response.json().get("detail", "Unknown error")
            print(f"   ⚠️  Chat endpoint error: {error}")
            return False
    except Exception as e:
        print(f"   ⚠️  Chat endpoint error: {e}")
        return False

def test_documents_endpoint(access_token: str):
    """Test documents endpoint (requires solicitor/admin role)"""
    print("\n7. Testing documents endpoint (requires solicitor/admin role)...")
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(
            f"{API_BASE_URL}/documents",
            headers=headers,
            params={"skip": 0, "limit": 10},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Documents endpoint accessible")
            print(f"      Documents count: {data.get('total', 0)}")
            return True
        elif response.status_code == 403:
            print(f"   ⚠️  Documents endpoint requires solicitor/admin role (403) - expected for public users")
            return True  # This is expected for public users
        elif response.status_code == 401:
            print(f"   ❌ Documents endpoint requires authentication (401)")
            return False
        else:
            error = response.json().get("detail", "Unknown error")
            print(f"   ⚠️  Documents endpoint error: {error}")
            return False
    except Exception as e:
        print(f"   ⚠️  Documents endpoint error: {e}")
        return False

def test_oauth_authorize(provider: str = "google"):
    """Test OAuth authorization URL"""
    print(f"\n8. Testing OAuth {provider} authorization...")
    try:
        response = requests.get(
            f"{API_BASE_URL}/auth/oauth/{provider}/authorize",
            params={"redirect_uri": "http://localhost:8501"},
            timeout=10,
            allow_redirects=False
        )
        
        if response.status_code == 302:
            auth_url = response.headers.get("Location")
            print(f"   ✅ OAuth authorization URL generated")
            print(f"      URL: {auth_url[:80]}..." if auth_url and len(auth_url) > 80 else f"      URL: {auth_url}")
            return True
        elif response.status_code == 200:
            data = response.json()
            auth_url = data.get("authorization_url")
            if auth_url:
                print(f"   ✅ OAuth authorization URL generated (JSON response)")
                return True
        else:
            error = response.json().get("detail", "Unknown error") if response.text else "OAuth not configured"
            print(f"   ⚠️  OAuth {provider} not configured: {error}")
            return True  # This is okay if OAuth is not configured
    except Exception as e:
        print(f"   ⚠️  OAuth authorization error: {e}")
        return True  # This is okay if OAuth is not configured

def main():
    print("=" * 60)
    print("FRONTEND AUTHENTICATION INTEGRATION TEST")
    print("=" * 60)
    
    # Test API health
    if not test_health():
        print("\n❌ API server is not running. Please start it first:")
        print("   uvicorn app.api.main:app --reload --port 8000")
        sys.exit(1)
    
    # Test registration
    token_data, test_email = test_register()
    if not token_data or not test_email:
        print("\n❌ Registration failed. Cannot continue with tests.")
        sys.exit(1)
    
    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token")
    
    # Test login
    login_token_data = test_login(test_email, "testpassword123")
    if not login_token_data:
        print("\n❌ Login failed.")
    
    # Test get profile
    user_profile = test_get_profile(access_token)
    if not user_profile:
        print("\n❌ Get profile failed.")
    
    # Test token refresh
    new_access_token = test_refresh_token(refresh_token)
    if not new_access_token:
        print("\n❌ Token refresh failed.")
    
    # Test protected endpoints
    if new_access_token:
        test_access_token = new_access_token
    else:
        test_access_token = access_token
    
    chat_success = test_chat_endpoint(test_access_token)
    docs_success = test_documents_endpoint(test_access_token)
    
    # Test OAuth (optional - may not be configured)
    test_oauth_authorize("google")
    test_oauth_authorize("github")
    test_oauth_authorize("microsoft")
    
    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    results = {
        "API Health": True,
        "Registration": token_data is not None,
        "Login": login_token_data is not None,
        "Get Profile": user_profile is not None,
        "Token Refresh": new_access_token is not None,
        "Chat Endpoint": chat_success,
        "Documents Endpoint": docs_success,
    }
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ All integration tests passed!")
        print("\n🚀 Frontend authentication is ready!")
        print("\nNext steps:")
        print("   1. Start Streamlit: streamlit run frontend/app.py")
        print("   2. Open http://localhost:8501 in your browser")
        print("   3. Test login/register flow")
        print("   4. Test role-based UI rendering")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

