#!/usr/bin/env python3
"""
Run Tests with Server Management
Automatically starts server, runs tests, and cleans up
"""

import subprocess
import time
import sys
import signal
import os
from pathlib import Path

def start_server():
    """Start the API server"""
    print("🚀 Starting API server...")
    proc = subprocess.Popen(
        ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path(__file__).parent.parent
    )
    
    # Wait for server to be ready
    print("⏳ Waiting for server to start...")
    for i in range(30):
        try:
            import requests
            response = requests.get("http://localhost:8000/api/v1/health", timeout=2)
            if response.status_code == 200:
                print("✅ Server is ready!")
                return proc
        except:
            pass
        time.sleep(1)
        if i % 5 == 0:
            print(f"   Still waiting... ({i}/30)")
    
    print("❌ Server failed to start in time")
    proc.terminate()
    return None

def run_tests(test_paths):
    """Run tests"""
    print(f"\n🧪 Running tests...")
    print("=" * 60)
    
    cmd = ["pytest"] + test_paths + ["-v", "--tb=short"]
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests with server")
    parser.add_argument("test_paths", nargs="*", default=["tests/"], help="Test paths to run")
    parser.add_argument("--no-server", action="store_true", help="Skip server start (assume running)")
    
    args = parser.parse_args()
    
    server_proc = None
    
    try:
        if not args.no_server:
            server_proc = start_server()
            if server_proc is None:
                print("❌ Failed to start server")
                return 1
        
        # Run tests
        success = run_tests(args.test_paths)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return 1
    finally:
        if server_proc:
            print("\n🛑 Stopping server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
            print("✅ Server stopped")

if __name__ == "__main__":
    exit(main())

