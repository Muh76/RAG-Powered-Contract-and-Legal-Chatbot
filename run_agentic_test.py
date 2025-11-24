#!/usr/bin/env python3
"""
Complete test script for Agentic RAG Legal Chatbot
Sets up environment and tests the agentic chat endpoint
"""

import os
import sys
import requests
import json
import time

# Set OpenAI API Key from environment variable
# Set this before running: export OPENAI_API_KEY="your-api-key-here"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
else:
    print("⚠️  Warning: OPENAI_API_KEY environment variable not set")
    print("   Set it with: export OPENAI_API_KEY='your-api-key-here'")

# Configuration
API_BASE_URL = "http://localhost:8000"
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


def check_server():
    """Check if server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print("✅ Server is running")
            print(f"   Status: {health.get('status')}")
            return True
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print(f"   Make sure the server is running on {API_BASE_URL}")
        return False


def register_or_login():
    """Register or login and get access token"""
    print_section("Authentication")
    
    # Try to register first
    try:
        response = requests.post(
            f"{AUTH_ENDPOINT}/register",
            json={
                "email": TEST_EMAIL,
                "password": TEST_PASSWORD,
                "full_name": TEST_NAME,
                "role": "public"
            },
            timeout=10
        )
        
        if response.status_code == 201:
            tokens = response.json()
            print(f"✅ User registered: {TEST_EMAIL}")
            return tokens.get("access_token")
        elif response.status_code == 400:
            # User might already exist, try to login
            print(f"⚠️  User might already exist, trying to login...")
        else:
            print(f"⚠️  Registration returned {response.status_code}, trying login...")
    except Exception as e:
        print(f"⚠️  Registration error: {e}, trying login...")
    
    # Try to login
    try:
        response = requests.post(
            f"{AUTH_ENDPOINT}/login",
            json={
                "email": TEST_EMAIL,
                "password": TEST_PASSWORD
            },
            timeout=10
        )
        
        if response.status_code == 200:
            tokens = response.json()
            print(f"✅ Login successful: {TEST_EMAIL}")
            return tokens.get("access_token")
        else:
            print(f"❌ Login failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Login error: {e}")
        return None


def test_agentic_chat(query: str, mode: str = "public", token: str = None):
    """Test agentic chat endpoint"""
    print(f"\n📝 Query: {query}")
    print(f"🎯 Mode: {mode}")
    
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    payload = {
        "query": query,
        "mode": mode,
        "chat_history": []
    }
    
    try:
        print("⏳ Sending request (this may take 30-60 seconds)...")
        start_time = time.time()
        response = requests.post(
            AGENTIC_CHAT_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=180
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Response received in {elapsed_time:.2f} seconds")
            print("\n" + "-" * 70)
            print("📄 ANSWER:")
            print("-" * 70)
            answer = result.get('answer', 'N/A')
            print(answer[:1000] + ("..." if len(answer) > 1000 else ""))
            
            tool_calls = result.get('tool_calls', [])
            print(f"\n🔧 TOOL CALLS: {len(tool_calls)}")
            for i, tool_call in enumerate(tool_calls, 1):
                print(f"  [{i}] {tool_call.get('tool', 'unknown')}")
                input_data = tool_call.get('input', {})
                if input_data:
                    print(f"      Input: {json.dumps(input_data, indent=8)}")
            
            print(f"\n📊 METRICS:")
            print(f"  - Iterations: {result.get('iterations', 0)}")
            print(f"  - Intermediate Steps: {result.get('intermediate_steps_count', 0)}")
            print(f"  - Confidence Score: {result.get('confidence_score', 0.0):.3f}")
            
            metrics = result.get('metrics', {})
            print(f"  - Total Time: {metrics.get('total_time_ms', 0.0):.2f} ms")
            print(f"  - Retrieval Time: {metrics.get('retrieval_time_ms', 0.0):.2f} ms")
            print(f"  - Generation Time: {metrics.get('generation_time_ms', 0.0):.2f} ms")
            
            safety = result.get('safety', {})
            print(f"\n🛡️ SAFETY:")
            print(f"  - Is Safe: {safety.get('is_safe', False)}")
            print(f"  - Flags: {safety.get('flags', [])}")
            
            return result
        elif response.status_code == 401:
            print(f"❌ Authentication failed (401)")
            print("   Response:", response.text[:200])
            return None
        elif response.status_code == 403:
            print(f"❌ Access forbidden (403)")
            print("   Response:", response.text[:200])
            return None
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Response: {response.text[:500]}")
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
    print_section("Agentic RAG Legal Chatbot - Complete Test")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"OpenAI API Key: {'✅ Set' if OPENAI_API_KEY else '❌ Not set'}")
    
    # Step 1: Check server
    if not check_server():
        print("\n❌ Cannot proceed - server is not running")
        print("   Start the server with: uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000")
        return
    
    # Step 2: Get authentication token
    token = register_or_login()
    if not token:
        print("\n❌ Cannot proceed without authentication token")
        return
    
    print(f"\n✅ Authentication token obtained")
    
    # Step 3: Test simple query
    print_section("Test 1: Simple Legal Query")
    result1 = test_agentic_chat(
        query="What are the key provisions of the Sale of Goods Act 1979?",
        mode="public",
        token=token
    )
    time.sleep(2)
    
    # Step 4: Complex multi-tool query
    print_section("Test 2: Complex Multi-Tool Query")
    result2 = test_agentic_chat(
        query="Compare the implied terms in the Sale of Goods Act 1979 with the Consumer Rights Act 2015",
        mode="public",
        token=token
    )
    
    # Summary
    print_section("✅ Test Summary")
    print(f"Test 1: {'✅ Passed' if result1 else '❌ Failed'}")
    print(f"Test 2: {'✅ Passed' if result2 else '❌ Failed'}")
    
    if result1 or result2:
        print("\n🎉 Agentic RAG chatbot is working!")
        print(f"\n📚 API Documentation: {API_BASE_URL}/docs")
        print(f"🔍 Health Check: {API_BASE_URL}/api/v1/health")


if __name__ == "__main__":
    main()

