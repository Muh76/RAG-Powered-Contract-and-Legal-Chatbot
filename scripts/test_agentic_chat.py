#!/usr/bin/env python3
"""
Test script for Agentic RAG Chatbot with LangChain
Tests the agentic chat endpoint and tool calling functionality
"""

import os
import sys
from pathlib import Path
import requests
import json
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_ENDPOINT = f"{API_BASE_URL}/api/v1/agentic-chat"


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(result: dict, title: str = "Result"):
    """Print formatted result"""
    print(f"\n{title}:")
    print("-" * 60)
    print(f"Answer: {result.get('answer', 'N/A')[:500]}...")
    print(f"\nTool Calls: {len(result.get('tool_calls', []))}")
    for i, tool_call in enumerate(result.get('tool_calls', []), 1):
        print(f"  [{i}] {tool_call.get('tool', 'unknown')}")
        print(f"      Input: {json.dumps(tool_call.get('input', {}), indent=6)[:200]}")
        print(f"      Result: {str(tool_call.get('result', ''))[:200]}...")
    print(f"\nIterations: {result.get('iterations', 0)}")
    print(f"Confidence Score: {result.get('confidence_score', 0.0):.3f}")
    print(f"Total Time: {result.get('metrics', {}).get('total_time_ms', 0.0):.2f} ms")
    print(f"Safety: {result.get('safety', {}).get('is_safe', False)}")


def test_agentic_chat(query: str, mode: str = "public", chat_history: list = None):
    """Test agentic chat endpoint"""
    try:
        payload = {
            "query": query,
            "mode": mode,
            "chat_history": chat_history or []
        }
        
        print(f"\nQuery: {query}")
        print(f"Mode: {mode}")
        
        response = requests.post(
            API_ENDPOINT,
            json=payload,
            timeout=120  # Longer timeout for agent execution
        )
        
        if response.status_code == 200:
            result = response.json()
            print_result(result)
            return result
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_agent_stats():
    """Test agent stats endpoint"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/agentic-chat/stats",
            timeout=10
        )
        
        if response.status_code == 200:
            stats = response.json()
            print_section("Agent Statistics")
            print(json.dumps(stats, indent=2))
            return stats
        else:
            print(f"Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    """Main test function"""
    print_section("Agentic RAG Chatbot Test Suite")
    print(f"API Endpoint: {API_ENDPOINT}")
    print(f"Base URL: {API_BASE_URL}")
    
    # Test 1: Simple query (should use tools)
    print_section("Test 1: Simple Legal Query")
    test_agentic_chat(
        query="What are the key provisions of the Sale of Goods Act 1979?",
        mode="public"
    )
    time.sleep(2)
    
    # Test 2: Query that requires statute lookup
    print_section("Test 2: Specific Statute Query")
    test_agentic_chat(
        query="Tell me about the Consumer Rights Act 2015",
        mode="public"
    )
    time.sleep(2)
    
    # Test 3: Complex query (might require multiple tools)
    print_section("Test 3: Complex Multi-Tool Query")
    test_agentic_chat(
        query="Compare the implied terms in the Sale of Goods Act 1979 with the Consumer Rights Act 2015",
        mode="public"
    )
    time.sleep(2)
    
    # Test 4: Solicitor mode
    print_section("Test 4: Solicitor Mode")
    test_agentic_chat(
        query="What are the statutory requirements for implied terms in commercial contracts?",
        mode="solicitor"
    )
    time.sleep(2)
    
    # Test 5: Conversation history
    print_section("Test 5: Conversation with History")
    history = [
        {"role": "user", "content": "What is the Sale of Goods Act 1979?"},
        {"role": "assistant", "content": "The Sale of Goods Act 1979 is UK legislation governing contracts for the sale of goods..."}
    ]
    test_agentic_chat(
        query="What about the Consumer Rights Act 2015?",
        mode="public",
        chat_history=history
    )
    time.sleep(2)
    
    # Test 6: Agent stats
    test_agent_stats()
    
    print_section("Test Suite Complete")


if __name__ == "__main__":
    main()

