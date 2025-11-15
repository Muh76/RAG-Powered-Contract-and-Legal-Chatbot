#!/usr/bin/env python3
# Legal Chatbot - Red Team Testing Automation
# Phase 2: Automated Adversarial Testing

import os
import sys
from pathlib import Path
import json
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from retrieval.red_team_tester import RedTeamTester, RedTeamTestResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_red_team_tester():
    """Test red team tester with guardrails service"""
    print("=" * 60)
    print("🛡️ Red Team Testing Automation")
    print("=" * 60)
    
    # Try to load guardrails service if available
    guardrails_service = None
    try:
        # Try importing guardrails (may not exist yet)
        from app.services.guardrails_service import GuardrailsService
        guardrails_service = GuardrailsService()
        print("✅ Guardrails service loaded")
    except (ImportError, AttributeError) as e:
        print(f"⚠️ Guardrails service not available: {e}")
        print("   Will use basic validation only")
    
    # Initialize red team tester
    tester = RedTeamTester(guardrails_service=guardrails_service)
    
    # Load test cases
    test_cases_path = project_root / "notebooks" / "phase2" / "eval" / "safety" / "safety_test_cases.json"
    if not test_cases_path.exists():
        print(f"⚠️ Test cases file not found: {test_cases_path}")
        print("   Using default test cases...")
    
    test_cases = tester.load_test_cases(test_cases_path if test_cases_path.exists() else None)
    
    print(f"\n📋 Loaded {len(test_cases)} test cases")
    print(f"   Categories: {set(tc.get('category', 'unknown') for tc in test_cases)}")
    
    # Run test suite
    print(f"\n🚀 Running test suite...")
    results = tester.run_test_suite(test_cases, use_rag=False)
    
    # Generate report
    print(f"\n📊 Generating test report...")
    report = tester.generate_report(results)
    
    # Display summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    summary = report["summary"]
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Pass Rate: {summary['pass_rate']:.1f}%")
    
    # By category
    print("\n📋 Results by Category:")
    for category, stats in report["by_category"].items():
        pass_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"   {category}: {stats['passed']}/{stats['total']} passed ({pass_rate:.1f}%)")
    
    # By risk level
    print("\n⚠️ Results by Risk Level:")
    for risk_level, stats in report["by_risk_level"].items():
        pass_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"   {risk_level}: {stats['passed']}/{stats['total']} passed ({pass_rate:.1f}%)")
    
    # Failed tests
    if report["failed_tests"]:
        print(f"\n❌ Failed Tests ({len(report['failed_tests'])}):")
        for test in report["failed_tests"][:10]:  # Show first 10
            print(f"   - {test['test_id']}: {test['category']}")
            print(f"     Query: {test['query'][:60]}...")
            print(f"     Expected: {test['expected_behavior']}, Got: {test['actual_behavior']}")
    
    # Save report
    report_path = project_root / "eval" / "red_team_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    tester.generate_report(results, output_path=report_path)
    print(f"\n💾 Report saved to: {report_path}")
    
    print("\n" + "=" * 60)
    print("✅ Red team testing completed!")
    print("=" * 60)
    
    return report


if __name__ == "__main__":
    try:
        report = test_red_team_tester()
        
        # Exit with error code if tests failed
        if report["summary"]["failed"] > 0:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        sys.exit(1)

