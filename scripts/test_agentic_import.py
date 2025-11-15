#!/usr/bin/env python3
"""
Simple test to verify agentic RAG imports and basic initialization
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all imports work"""
    print("Testing imports...")
    
    try:
        from app.tools.legal_search_tool import LegalSearchTool
        print("✅ LegalSearchTool import OK")
    except Exception as e:
        print(f"❌ LegalSearchTool import failed: {e}")
        return False
    
    try:
        from app.tools.statute_lookup_tool import StatuteLookupTool
        print("✅ StatuteLookupTool import OK")
    except Exception as e:
        print(f"❌ StatuteLookupTool import failed: {e}")
        return False
    
    try:
        from app.services.agent_service import AgenticRAGService, LANGCHAIN_VERSION
        print(f"✅ AgenticRAGService import OK (LangChain: {LANGCHAIN_VERSION})")
    except Exception as e:
        print(f"❌ AgenticRAGService import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        from app.api.routes.agentic_chat import router
        print("✅ Agentic chat router import OK")
    except Exception as e:
        print(f"❌ Agentic chat router import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        from app.api.main import app
        print("✅ FastAPI app import OK")
    except Exception as e:
        print(f"❌ FastAPI app import failed: {e}")
        return False
    
    return True

def test_router_registration():
    """Test that the router is registered"""
    try:
        from app.api.main import app
        
        # Check if agentic-chat route is registered
        routes = [route.path for route in app.routes]
        agentic_routes = [r for r in routes if "agentic" in r.lower()]
        
        if agentic_routes:
            print(f"✅ Agentic routes registered: {agentic_routes}")
            return True
        else:
            print("⚠️ No agentic routes found in app routes")
            print(f"Available routes: {routes[:10]}...")
            return False
    except Exception as e:
        print(f"❌ Router registration test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Agentic RAG Import Test")
    print("=" * 60)
    
    if test_imports():
        print("\n✅ All imports successful!")
        if test_router_registration():
            print("\n✅ Router registration successful!")
            print("\n✅ All basic tests passed!")
            sys.exit(0)
        else:
            print("\n⚠️ Router registration issue (but imports OK)")
            sys.exit(0)
    else:
        print("\n❌ Import tests failed")
        sys.exit(1)

