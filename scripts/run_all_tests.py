#!/usr/bin/env python3
"""
Comprehensive Test Runner
Phase 4.1: Run All Test Suites
"""

import sys
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any


class TestRunner:
    """Run all test suites"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results = {
            "start_time": time.time(),
            "tests": []
        }
    
    def run_pytest_suite(self, test_path: str, test_name: str) -> Dict[str, Any]:
        """Run a pytest test suite"""
        print(f"\n{'='*60}")
        print(f"🧪 Running: {test_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                ["pytest", test_path, "-v", "--tb=short"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            duration = time.time() - start_time
            
            test_result = {
                "name": test_name,
                "path": test_path,
                "success": result.returncode == 0,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
            if result.returncode == 0:
                print(f"✅ {test_name} passed ({duration:.2f}s)")
            else:
                print(f"❌ {test_name} failed ({duration:.2f}s)")
                print(result.stdout[:500])  # Print first 500 chars
                if result.stderr:
                    print(result.stderr[:500])
            
            return test_result
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"⏱️ {test_name} timed out after {duration:.2f}s")
            return {
                "name": test_name,
                "path": test_path,
                "success": False,
                "duration": duration,
                "error": "Timeout"
            }
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ {test_name} error: {e}")
            return {
                "name": test_name,
                "path": test_path,
                "success": False,
                "duration": duration,
                "error": str(e)
            }
    
    def run_script(self, script_path: str, script_name: str, *args) -> Dict[str, Any]:
        """Run a Python script"""
        print(f"\n{'='*60}")
        print(f"🚀 Running: {script_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                ["python", script_path] + list(args),
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600
            )
            duration = time.time() - start_time
            
            test_result = {
                "name": script_name,
                "path": script_path,
                "success": result.returncode == 0,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
            if result.returncode == 0:
                print(f"✅ {script_name} completed ({duration:.2f}s)")
            else:
                print(f"❌ {script_name} failed ({duration:.2f}s)")
                print(result.stdout[:500])
                if result.stderr:
                    print(result.stderr[:500])
            
            return test_result
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ {script_name} error: {e}")
            return {
                "name": script_name,
                "path": script_path,
                "success": False,
                "duration": duration,
                "error": str(e)
            }
    
    def run_all_tests(self):
        """Run all test suites"""
        print("🧪 Comprehensive Test Suite Runner")
        print("="*60)
        print(f"Project Root: {self.project_root}")
        
        # Test suites to run
        test_suites = [
            # E2E Tests
            ("tests/e2e/test_all_endpoints.py", "E2E: All Endpoints"),
            ("tests/e2e/test_error_scenarios.py", "E2E: Error Scenarios"),
            ("tests/e2e/test_load.py", "E2E: Load Testing"),
            ("tests/e2e/test_regression.py", "E2E: Regression Tests"),
            
            # Integration Tests
            ("tests/integration/test_service_integration.py", "Integration: Service Integration"),
            
            # Unit Tests
            ("tests/unit/test_basic.py", "Unit: Basic Tests"),
            
            # API Integration
            ("tests/test_api_integration.py", "Integration: API Integration"),
        ]
        
        # Run pytest suites
        for test_path, test_name in test_suites:
            if Path(self.project_root / test_path).exists():
                result = self.run_pytest_suite(test_path, test_name)
                self.results["tests"].append(result)
            else:
                print(f"⚠️ Test file not found: {test_path}")
        
        # Run performance benchmark (optional)
        perf_script = "scripts/test_performance_benchmark.py"
        if Path(self.project_root / perf_script).exists():
            result = self.run_script(perf_script, "Performance Benchmark", "--iterations", "3")
            self.results["tests"].append(result)
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print test summary"""
        print(f"\n{'='*60}")
        print("📊 Test Summary")
        print(f"{'='*60}")
        
        total_tests = len(self.results["tests"])
        successful = sum(1 for t in self.results["tests"] if t["success"])
        failed = total_tests - successful
        total_duration = sum(t["duration"] for t in self.results["tests"])
        
        print(f"\nTotal Test Suites: {total_tests}")
        print(f"✅ Passed: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"⏱️ Total Duration: {total_duration:.2f}s")
        
        if failed > 0:
            print(f"\n❌ Failed Tests:")
            for test in self.results["tests"]:
                if not test["success"]:
                    print(f"  - {test['name']}")
        
        # Overall result
        self.results["end_time"] = time.time()
        self.results["total_duration"] = self.results["end_time"] - self.results["start_time"]
        self.results["success_rate"] = (successful / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n🎯 Success Rate: {self.results['success_rate']:.1f}%")
        
        if self.results["success_rate"] == 100:
            print("🎉 All tests passed!")
        elif self.results["success_rate"] >= 80:
            print("⚠️ Some tests failed, but overall status is acceptable")
        else:
            print("❌ Multiple test failures detected")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Test Runner")
    parser.add_argument("--suite", type=str, help="Run specific test suite")
    parser.add_argument("--skip-perf", action="store_true", help="Skip performance benchmarks")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.suite:
        # Run specific suite
        if Path(runner.project_root / args.suite).exists():
            result = runner.run_pytest_suite(args.suite, args.suite)
            runner.results["tests"].append(result)
            runner._print_summary()
        else:
            print(f"❌ Test file not found: {args.suite}")
            return 1
    else:
        # Run all tests
        runner.run_all_tests()
    
    # Exit code based on results
    if runner.results["tests"]:
        all_passed = all(t["success"] for t in runner.results["tests"])
        return 0 if all_passed else 1
    return 1


if __name__ == "__main__":
    exit(main())

