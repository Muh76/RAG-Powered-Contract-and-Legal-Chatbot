#!/usr/bin/env python3
"""
Test script for Agentic RAG Legal Chatbot
Handles authentication and tests the agentic chat endpoint
"""

import os
import sys
import requests
import json
import time
from pathlib import Path

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
AUTH_ENDPOINT = f"{API_BASE_URL}/api/v1/auth"
AGENTIC_CHAT_ENDPOINT = f"{API_BASE_URL}/api/v1/agentic-chat"

# Test user credentials
TEST_EMAIL = "test_agentic@example.com"
TEST_PASSWORD = "test_password_123"
TEST_NAME = "Test User"


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(result: dict, title: str = "Result"):
    """Print formatted result"""
    print(f"\n{title}:")
    print("-" * 70)
    
    answer = result.get('answer', 'N/A')
    print(f"Answer ({len(answer)} chars):")
    print(answer[:800] + ("..." if len(answer) > 800 else ""))
    
    tool_calls = result.get('tool_calls', [])
    print(f"\n🔧 Tool Calls: {len(tool_calls)}")
    for i, tool_call in enumerate(tool_calls, 1):
        print(f"\n  [{i}] Tool: {tool_call.get('tool', 'unknown')}")
        print(f"      Input: {json.dumps(tool_call.get('input', {}), indent=8)}")
        result_text = str(tool_call.get('result', ''))
        print(f"      Result: {result_text[:300]}..." if len(result_text) > 300 else f"      Result: {result_text}")
    
    print(f"\n📊 Metrics:")
    metrics = result.get('metrics', {})
    print(f"  - Iterations: {result.get('iterations', 0)}")
    print(f"  - Intermediate Steps: {result.get('intermediate_steps_count', 0)}")
    print(f"  - Confidence Score: {result.get('confidence_score', 0.0):.3f}")
    print(f"  - Total Time: {metrics.get('total_time_ms', 0.0):.2f} ms")
    print(f"  - Retrieval Time: {metrics.get('retrieval_time_ms', 0.0):.2f} ms")
    print(f"  - Generation Time: {metrics.get('generation_time_ms', 0.0):.2f} ms")
    
    safety = result.get('safety', {})
    print(f"\n🛡️ Safety:")
    print(f"  - Is Safe: {safety.get('is_safe', False)}")
    print(f"  - Flags: {safety.get('flags', [])}")
    print(f"  - Confidence: {safety.get('confidence', 0.0):.3f}")


def register_user(email: str, password: str, full_name: str, role: str = "public"):
    """Register a new user"""
    try:
        response = requests.post(
            f"{AUTH_ENDPOINT}/register",
            json={
                "email": email,
                "password": password,
                "full_name": full_name,
                "role": role
            },
            timeout=10
        )
        
        if response.status_code == 201:
            tokens = response.json()
            print(f"✅ User registered successfully: {email}")
            return tokens.get("access_token")
        elif response.status_code == 400:
            # User might already exist, try to login
            print(f"⚠️  User might already exist, trying to login...")
            return login_user(email, password)
        else:
            print(f"❌ Registration failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return None


def login_user(email: str, password: str):
    """Login and get access token"""
    try:
        response = requests.post(
            f"{AUTH_ENDPOINT}/login",
            json={
                "email": email,
                "password": password
            },
            timeout=10
        )
        
        if response.status_code == 200:
            tokens = response.json()
            print(f"✅ Login successful: {email}")
            return tokens.get("access_token")
        else:
            print(f"❌ Login failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Login error: {e}")
        return None


def get_auth_token():
    """Get authentication token (register or login)"""
    print_section("Authentication")
    
    # Try to register first
    token = register_user(TEST_EMAIL, TEST_PASSWORD, TEST_NAME, "public")
    
    if not token:
        # If registration fails, try login
        token = login_user(TEST_EMAIL, TEST_PASSWORD)
    
    if token:
        print(f"✅ Authentication token obtained")
        return token
    else:
        print("❌ Failed to get authentication token")
        return None


def test_agentic_chat(query: str, mode: str = "public", chat_history: list = None, token: str = None):
    """Test agentic chat endpoint"""
    try:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        payload = {
            "query": query,
            "mode": mode,
            "chat_history": chat_history or []
        }
        
        print(f"\n📝 Query: {query}")
        print(f"🎯 Mode: {mode}")
        
        start_time = time.time()
        response = requests.post(
            AGENTIC_CHAT_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=180  # Longer timeout for agent execution
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Response received in {elapsed_time:.2f} seconds")
            print_result(result)
            return result
        elif response.status_code == 401:
            print(f"❌ Authentication failed (401)")
            print("Response:", response.text)
            return None
        elif response.status_code == 403:
            print(f"❌ Access forbidden (403) - Role might not have permission")
            print("Response:", response.text)
            return None
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"❌ Request timeout (>180s)")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main test function"""
    print_section("Agentic RAG Legal Chatbot Test Suite")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Agentic Chat Endpoint: {AGENTIC_CHAT_ENDPOINT}")
    
    # Step 1: Get authentication token
    token = get_auth_token()
    if not token:
        print("\n❌ Cannot proceed without authentication token")
        print("Please ensure the database is set up and running")
        return
    
    # Step 2: Test simple query
    print_section("Test 1: Simple Legal Query")
    test_agentic_chat(
        query="What are the key provisions of the Sale of Goods Act 1979?",
        mode="public",
        token=token
    )
    time.sleep(2)
    
    # Step 3: Specific statute query
    print_section("Test 2: Specific Statute Query")
    test_agentic_chat(
        query="Tell me about the Consumer Rights Act 2015",
        mode="public",
        token=token
    )
    time.sleep(2)
    
    # Step 4: Complex multi-tool query
    print_section("Test 3: Complex Multi-Tool Query")
    test_agentic_chat(
        query="Compare the implied terms in the Sale of Goods Act 1979 with the Consumer Rights Act 2015",
        mode="public",
        token=token
    )
    time.sleep(2)
    
    # Step 5: Conversation with history
    print_section("Test 4: Conversation with History")
    history = [
        {"role": "user", "content": "What is the Sale of Goods Act 1979?"},
        {"role": "assistant", "content": "The Sale of Goods Act 1979 is UK legislation governing contracts for the sale of goods."}
    ]
    test_agentic_chat(
        query="What about the Consumer Rights Act 2015?",
        mode="public",
        chat_history=history,
        token=token
    )
    
    print_section("✅ Test Suite Complete")
    print("\n💡 Tips:")
    print("  - Check the tool_calls to see which tools the agent used")
    print("  - Multiple iterations indicate multi-step reasoning")
    print("  - Complex queries trigger multiple tool calls automatically")
    print(f"\n📚 API Documentation: {API_BASE_URL}/docs")
    print(f"🔍 Health Check: {API_BASE_URL}/api/v1/health")


if __name__ == "__main__":
    main()

