#!/usr/bin/env python3
"""
Verification Script for Phase 4.2: Monitoring and Observability
Tests all monitoring features to ensure they work seamlessly
"""

import sys
import os
import time
import json
import requests
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


class MonitoringVerifier:
    """Verifies all monitoring and observability features"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    def log(self, message: str, status: str = "INFO"):
        """Log a message with status"""
        symbols = {
            "SUCCESS": f"{GREEN}✅{RESET}",
            "ERROR": f"{RED}❌{RESET}",
            "WARNING": f"{YELLOW}⚠️{RESET}",
            "INFO": f"{BLUE}ℹ️{RESET}",
        }
        symbol = symbols.get(status, "ℹ️")
        print(f"{symbol} {message}")
        self.results.append({"message": message, "status": status, "timestamp": datetime.now().isoformat()})
    
    def test_health_endpoints(self) -> bool:
        """Test all health check endpoints"""
        self.log("Testing Health Check Endpoints", "INFO")
        print()
        
        tests = [
            ("/api/v1/health", "Basic health check"),
            ("/api/v1/health/detailed", "Detailed health check"),
            ("/api/v1/health/live", "Liveness probe"),
            ("/api/v1/health/ready", "Readiness probe"),
        ]
        
        all_passed = True
        for endpoint, description in tests:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                if response.status_code == 200 or response.status_code == 503:
                    data = response.json()
                    if "status" in data or "services" in data:
                        self.log(f"  {description}: PASSED (Status: {response.status_code})", "SUCCESS")
                    else:
                        self.log(f"  {description}: FAILED (Missing status/services)", "ERROR")
                        all_passed = False
                else:
                    self.log(f"  {description}: FAILED (Status: {response.status_code})", "ERROR")
                    all_passed = False
            except requests.exceptions.ConnectionError:
                self.log(f"  {description}: FAILED (Cannot connect to server)", "ERROR")
                self.log("    Make sure the server is running: uvicorn app.api.main:app --reload", "WARNING")
                all_passed = False
            except Exception as e:
                self.log(f"  {description}: FAILED ({str(e)})", "ERROR")
                all_passed = False
        
        return all_passed
    
    def test_metrics_endpoints(self) -> bool:
        """Test all metrics endpoints"""
        self.log("Testing Metrics Endpoints", "INFO")
        print()
        
        tests = [
            ("/api/v1/metrics", "All metrics"),
            ("/api/v1/metrics/summary", "Summary metrics"),
            ("/api/v1/metrics/endpoints", "Endpoint metrics"),
            ("/api/v1/metrics/tools", "Tool metrics"),
            ("/api/v1/metrics/system", "System metrics"),
        ]
        
        all_passed = True
        for endpoint, description in tests:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict) and len(data) > 0:
                        self.log(f"  {description}: PASSED (Returned {len(data)} keys)", "SUCCESS")
                    else:
                        self.log(f"  {description}: FAILED (Empty or invalid response)", "ERROR")
                        all_passed = False
                else:
                    self.log(f"  {description}: FAILED (Status: {response.status_code})", "ERROR")
                    all_passed = False
            except requests.exceptions.ConnectionError:
                self.log(f"  {description}: FAILED (Cannot connect to server)", "ERROR")
                all_passed = False
            except Exception as e:
                self.log(f"  {description}: FAILED ({str(e)})", "ERROR")
                all_passed = False
        
        return all_passed
    
    def test_logging_functionality(self) -> bool:
        """Test logging functionality"""
        self.log("Testing Logging Functionality", "INFO")
        print()
        
        # Check if logs directory exists
        log_dir = Path("logs")
        log_file = log_dir / "legal_chatbot.log"
        
        if not log_dir.exists():
            self.log("  Logs directory not found: Creating...", "WARNING")
            log_dir.mkdir(parents=True, exist_ok=True)
        
        # Check log file format
        if log_file.exists():
            try:
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        # Try to parse as JSON
                        try:
                            log_data = json.loads(last_line)
                            if "timestamp" in log_data and "level" in log_data:
                                self.log("  JSON log format: PASSED", "SUCCESS")
                            else:
                                self.log("  JSON log format: FAILED (Missing required fields)", "WARNING")
                        except json.JSONDecodeError:
                            # Not JSON, check if it's standard format
                            if "|" in last_line:
                                self.log("  Standard log format: PASSED", "SUCCESS")
                            else:
                                self.log("  Log format: UNKNOWN", "WARNING")
                    else:
                        self.log("  Log file is empty: No logs yet", "WARNING")
            except Exception as e:
                self.log(f"  Log file check: FAILED ({str(e)})", "ERROR")
                return False
        else:
            self.log("  Log file not found: Will be created on first request", "WARNING")
        
        # Test if logging module can be imported and initialized
        try:
            from app.core.logging import setup_logging
            logger = setup_logging()
            self.log("  Logging module initialization: PASSED", "SUCCESS")
            return True
        except Exception as e:
            self.log(f"  Logging module initialization: FAILED ({str(e)})", "ERROR")
            return False
    
    def test_metrics_collection(self) -> bool:
        """Test metrics collection functionality"""
        self.log("Testing Metrics Collection", "INFO")
        print()
        
        try:
            from app.core.metrics import metrics_collector, SystemMetrics
            
            # Test API metrics recording
            metrics_collector.record_api_request(
                endpoint="/api/v1/test",
                method="GET",
                response_time_ms=100.5,
                status_code=200,
            )
            
            # Test tool usage recording
            metrics_collector.record_tool_usage(
                tool_name="test_tool",
                execution_time_ms=50.0,
                success=True,
            )
            
            # Get metrics
            summary = metrics_collector.get_summary_metrics()
            if summary and "total_requests" in summary:
                self.log("  API metrics collection: PASSED", "SUCCESS")
            else:
                self.log("  API metrics collection: FAILED (No summary data)", "ERROR")
                return False
            
            # Test system metrics
            system_metrics = SystemMetrics.get_all_metrics()
            if system_metrics and "cpu" in system_metrics and "memory" in system_metrics:
                self.log("  System metrics collection: PASSED", "SUCCESS")
                self.log(f"    CPU: {system_metrics['cpu'].get('cpu_percent', 'N/A')}%", "INFO")
                self.log(f"    Memory: {system_metrics['memory'].get('memory_percent', 'N/A')}%", "INFO")
            else:
                self.log("  System metrics collection: FAILED (Missing metrics)", "ERROR")
                return False
            
            return True
        except Exception as e:
            self.log(f"  Metrics collection: FAILED ({str(e)})", "ERROR")
            return False
    
    def test_health_checker(self) -> bool:
        """Test health checker functionality"""
        self.log("Testing Health Checker", "INFO")
        print()
        
        try:
            from app.core.health_checker import health_checker
            import asyncio
            
            async def test_checks():
                dependencies = await health_checker.check_all_dependencies()
                return dependencies
            
            dependencies = asyncio.run(test_checks())
            
            if dependencies and isinstance(dependencies, dict):
                self.log("  Health checker initialization: PASSED", "SUCCESS")
                for service, status in dependencies.items():
                    service_status = status.get("status", "unknown")
                    if service_status in ["healthy", "unhealthy", "unknown"]:
                        self.log(f"    {service}: {service_status}", "INFO")
                    else:
                        self.log(f"    {service}: UNKNOWN STATUS", "WARNING")
                return True
            else:
                self.log("  Health checker: FAILED (Invalid response)", "ERROR")
                return False
        except Exception as e:
            self.log(f"  Health checker: FAILED ({str(e)})", "ERROR")
            return False
    
    def test_middleware(self) -> bool:
        """Test middleware functionality"""
        self.log("Testing Middleware", "INFO")
        print()
        
        try:
            from app.core.middleware import RequestResponseLoggingMiddleware, ErrorTrackingMiddleware
            
            # Check if middleware classes exist
            if RequestResponseLoggingMiddleware and ErrorTrackingMiddleware:
                self.log("  Middleware classes: PASSED", "SUCCESS")
                return True
            else:
                self.log("  Middleware classes: FAILED", "ERROR")
                return False
        except Exception as e:
            self.log(f"  Middleware: FAILED ({str(e)})", "ERROR")
            return False
    
    def test_request_response_headers(self) -> bool:
        """Test request/response headers from actual API calls"""
        self.log("Testing Request/Response Headers", "INFO")
        print()
        
        try:
            response = requests.get(f"{self.base_url}/api/v1/health", timeout=10)
            
            # Check for custom headers
            headers_to_check = ["X-Request-ID", "X-Process-Time"]
            found_headers = []
            
            for header in headers_to_check:
                if header in response.headers:
                    found_headers.append(header)
                    self.log(f"  {header}: FOUND ({response.headers[header]})", "SUCCESS")
                else:
                    self.log(f"  {header}: NOT FOUND", "WARNING")
            
            if len(found_headers) > 0:
                return True
            else:
                self.log("  No custom headers found: Middleware may not be active", "WARNING")
                return False
        except requests.exceptions.ConnectionError:
            self.log("  Cannot test headers: Server not running", "ERROR")
            return False
        except Exception as e:
            self.log(f"  Header test: FAILED ({str(e)})", "ERROR")
            return False
    
    def test_integration(self) -> bool:
        """Test full integration - make actual API calls and verify logging/metrics"""
        self.log("Testing Full Integration", "INFO")
        print()
        
        try:
            # Make a test API call
            response = requests.get(f"{self.base_url}/api/v1/health", timeout=10)
            
            if response.status_code == 200:
                self.log("  API call successful: PASSED", "SUCCESS")
                
                # Wait a bit for metrics to be recorded
                time.sleep(0.5)
                
                # Check if metrics were recorded
                try:
                    metrics_response = requests.get(f"{self.base_url}/api/v1/metrics/summary", timeout=10)
                    if metrics_response.status_code == 200:
                        metrics_data = metrics_response.json()
                        if metrics_data.get("total_requests", 0) > 0:
                            self.log("  Metrics recorded: PASSED", "SUCCESS")
                            self.log(f"    Total requests: {metrics_data.get('total_requests', 0)}", "INFO")
                        else:
                            self.log("  Metrics recorded: No requests tracked yet", "WARNING")
                    else:
                        self.log("  Metrics recorded: FAILED (Cannot fetch metrics)", "ERROR")
                except Exception as e:
                    self.log(f"  Metrics check: FAILED ({str(e)})", "WARNING")
                
                return True
            else:
                self.log(f"  API call: FAILED (Status: {response.status_code})", "ERROR")
                return False
        except requests.exceptions.ConnectionError:
            self.log("  Integration test: Cannot connect to server", "ERROR")
            self.log("    Start server with: uvicorn app.api.main:app --reload", "WARNING")
            return False
        except Exception as e:
            self.log(f"  Integration test: FAILED ({str(e)})", "ERROR")
            return False
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate verification report"""
        report = {
            "verification_timestamp": datetime.now().isoformat(),
            "base_url": self.base_url,
            "results": self.results,
            "summary": {
                "total_tests": len([r for r in self.results if r["status"] in ["SUCCESS", "ERROR"]]),
                "passed": len([r for r in self.results if r["status"] == "SUCCESS"]),
                "failed": len([r for r in self.results if r["status"] == "ERROR"]),
                "warnings": len([r for r in self.results if r["status"] == "WARNING"]),
            }
        }
        return report
    
    def run_all_tests(self) -> bool:
        """Run all verification tests"""
        print(f"{BOLD}{BLUE}{'='*60}{RESET}")
        print(f"{BOLD}{BLUE}Phase 4.2: Monitoring & Observability Verification{RESET}")
        print(f"{BOLD}{BLUE}{'='*60}{RESET}")
        print()
        
        tests = [
            ("Health Checker", self.test_health_checker),
            ("Middleware", self.test_middleware),
            ("Logging Functionality", self.test_logging_functionality),
            ("Metrics Collection", self.test_metrics_collection),
            ("Health Endpoints", self.test_health_endpoints),
            ("Metrics Endpoints", self.test_metrics_endpoints),
            ("Request/Response Headers", self.test_request_response_headers),
            ("Full Integration", self.test_integration),
        ]
        
        results = {}
        for test_name, test_func in tests:
            print()
            try:
                results[test_name] = test_func()
            except Exception as e:
                self.log(f"{test_name}: FAILED with exception ({str(e)})", "ERROR")
                results[test_name] = False
        
        # Generate summary
        print()
        print(f"{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}Verification Summary{RESET}")
        print(f"{BOLD}{'='*60}{RESET}")
        print()
        
        total = len(results)
        passed = sum(1 for v in results.values() if v)
        failed = total - passed
        
        for test_name, result in results.items():
            status = f"{GREEN}✅ PASSED{RESET}" if result else f"{RED}❌ FAILED{RESET}"
            print(f"  {test_name}: {status}")
        
        print()
        print(f"{BOLD}Overall:{RESET}")
        print(f"  Total Tests: {total}")
        print(f"  {GREEN}Passed: {passed}{RESET}")
        print(f"  {RED}Failed: {failed}{RESET}")
        print()
        
        if passed == total:
            print(f"{GREEN}{BOLD}✅ All tests passed! Monitoring is working seamlessly.{RESET}")
            return True
        else:
            print(f"{YELLOW}{BOLD}⚠️  Some tests failed. Please review the output above.{RESET}")
            return False


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Phase 4.2 Monitoring Implementation")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API server")
    parser.add_argument("--report", help="Save report to JSON file")
    
    args = parser.parse_args()
    
    verifier = MonitoringVerifier(base_url=args.url)
    success = verifier.run_all_tests()
    
    if args.report:
        report = verifier.generate_report()
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved to: {args.report}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

