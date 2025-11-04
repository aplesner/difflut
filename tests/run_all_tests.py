#!/usr/bin/env python3
"""
Comprehensive DiffLUT Test Suite Runner.
Runs all tests in the correct order and provides summary.
"""

import sys
import subprocess
import time
from pathlib import Path


def run_test(test_file: str, test_name: str) -> tuple:
    """
    Run a single test file.
    
    Args:
        test_file: Name of test file (without .py)
        test_name: Display name of test
        
    Returns:
        Tuple of (passed: bool, runtime: float)
    """
    print(f"\n{'='*70}")
    print(f"  Running: {test_name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    result = subprocess.run(
        [sys.executable, f"{test_file}.py"],
        cwd=str(Path(__file__).parent),
        capture_output=False
    )
    
    runtime = time.time() - start_time
    passed = result.returncode == 0
    
    return passed, runtime


def main():
    """Run all tests in order."""
    print("\n" + "="*70)
    print("  DIFFLUT COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        ("test_utils", "Test Utilities Module"),
        ("test_registry_validation", "Registry Validation Tests"),
        ("test_nodes_forward_pass", "Node Forward Pass Tests"),
        ("test_layers_forward_pass", "Layer Forward Pass Tests"),
        ("test_encoders_forward_pass", "Encoder Forward Pass Tests"),
        ("test_utils_modules", "Utility Modules Tests (GroupSum)"),
        ("test_all_nodes_training", "Node Training Performance Tests"),
    ]
    
    results = {}
    total_runtime = 0
    
    for test_file, test_name in tests:
        try:
            passed, runtime = run_test(test_file, test_name)
            results[test_name] = (passed, runtime)
            total_runtime += runtime
        except Exception as e:
            print(f"\n✗ FAILED TO RUN: {test_name}")
            print(f"  Error: {e}")
            results[test_name] = (False, 0)
    
    # Print summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for passed, _ in results.values() if passed)
    failed_count = len(results) - passed_count
    
    for test_name, (passed, runtime) in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name:50s} ({runtime:6.2f}s)")
    
    print(f"\n  Total: {passed_count}/{len(results)} test suites passed ({total_runtime:.2f}s)")
    
    if failed_count > 0:
        print(f"\n⚠ {failed_count} test suite(s) failed!")
        sys.exit(1)
    else:
        print("\n✓ All test suites passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
