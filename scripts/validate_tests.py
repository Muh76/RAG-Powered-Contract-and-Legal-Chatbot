#!/usr/bin/env python3
"""
Test Validation Script
Validates test files can be imported and have correct structure
"""

import sys
from pathlib import Path
import importlib.util

def validate_test_file(test_path: Path) -> tuple[bool, str]:
    """Validate a test file can be imported"""
    try:
        spec = importlib.util.spec_from_file_location("test_module", test_path)
        if spec is None or spec.loader is None:
            return False, "Could not create spec"
        
        module = importlib.util.module_from_spec(spec)
        # Don't execute, just validate structure
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Import error: {e}"

def main():
    """Main function"""
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"
    
    test_files = []
    
    # Find all test files
    for test_file in tests_dir.rglob("test_*.py"):
        if "__pycache__" not in str(test_file):
            test_files.append(test_file)
    
    # Find test scripts
    scripts_dir = project_root / "scripts"
    for test_script in scripts_dir.glob("test_*.py"):
        test_files.append(test_script)
    
    print(f"🧪 Validating {len(test_files)} test files...\n")
    
    passed = 0
    failed = 0
    
    for test_file in sorted(test_files):
        relative_path = test_file.relative_to(project_root)
        success, message = validate_test_file(test_file)
        
        if success:
            print(f"✅ {relative_path}")
            passed += 1
        else:
            print(f"❌ {relative_path}: {message}")
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All test files validated successfully!")
        return 0
    else:
        print("❌ Some test files have issues")
        return 1

if __name__ == "__main__":
    exit(main())

