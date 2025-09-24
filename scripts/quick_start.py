#!/usr/bin/env python3
"""
Quick Start Script for Legal Chatbot Phase 1
"""

import subprocess
import sys
import time
import requests
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    try:
        import fastapi
        import streamlit
        import openai
        import faiss
        import sklearn
        print("✅ All dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def check_openai_key():
    """Check if OpenAI API key is set"""
    print("🔑 Checking OpenAI API key...")
    
    if os.getenv('OPENAI_API_KEY'):
        print("✅ OpenAI API key found")
        return True
    else:
        print("❌ OpenAI API key not found")
        print("Set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-key-here'")
        return False

def start_fastapi():
    """Start the FastAPI server"""
    print("🚀 Starting FastAPI server...")
    
    try:
        # Start FastAPI in background
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "app.api.main:app", 
            "--reload", 
            "--port", "8000"
        ])
        
        # Wait for server to start
        for i in range(30):
            try:
                response = requests.get("http://localhost:8000/health", timeout=1)
                if response.status_code == 200:
                    print("✅ FastAPI server started successfully")
                    return process
            except requests.exceptions.RequestException:
                time.sleep(1)
        
        print("❌ FastAPI server failed to start")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"❌ Error starting FastAPI: {e}")
        return None

def start_streamlit():
    """Start the Streamlit UI"""
    print("🎨 Starting Streamlit UI...")
    
    try:
        # Start Streamlit
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", 
            "run", "frontend/app.py",
            "--server.port", "8501"
        ])
        
        # Wait a bit for Streamlit to start
        time.sleep(3)
        print("✅ Streamlit UI started successfully")
        print("🌐 Open: http://localhost:8501")
        return process
        
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        return None

def run_tests():
    """Run API integration tests"""
    print("🧪 Running API tests...")
    
    try:
        result = subprocess.run([
            sys.executable, "tests/test_api_integration.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All tests passed")
            print(result.stdout)
        else:
            print("❌ Some tests failed")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def main():
    """Main quick start function"""
    print("⚖️ Legal Chatbot Phase 1 - Quick Start")
    print("=" * 50)
    
    # Check prerequisites
    if not check_dependencies():
        return
    
    if not check_openai_key():
        return
    
    print("\n🎯 Starting Legal Chatbot...")
    
    # Start FastAPI
    api_process = start_fastapi()
    if not api_process:
        return
    
    # Wait a moment for API to be ready
    time.sleep(2)
    
    # Run tests
    run_tests()
    
    # Start Streamlit
    streamlit_process = start_streamlit()
    if not streamlit_process:
        api_process.terminate()
        return
    
    print("\n🎉 Legal Chatbot is running!")
    print("📱 Streamlit UI: http://localhost:8501")
    print("🔧 FastAPI Docs: http://localhost:8000/docs")
    print("💡 Press Ctrl+C to stop all services")
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        if api_process:
            api_process.terminate()
        if streamlit_process:
            streamlit_process.terminate()
        print("✅ All services stopped")

if __name__ == "__main__":
    main()
