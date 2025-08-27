#!/usr/bin/env python3
"""
Test runner script for subgroup fairness experiments

Usage:
    python run_tests.py [options]

Examples:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run only unit tests  
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --performance      # Run only performance tests
    python run_tests.py --fast             # Skip slow tests
    python run_tests.py --coverage         # Run with coverage report
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_pytest_command(args_list):
    """Run pytest with given arguments"""
    cmd = ['python', '-m', 'pytest'] + args_list
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Run subgroup fairness tests")
    
    # Test category selection
    parser.add_argument('--unit', action='store_true', 
                       help='Run unit tests only')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests only')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance benchmark tests only')
    
    # Test filtering options
    parser.add_argument('--fast', action='store_true',
                       help='Skip slow tests (exclude -m slow)')
    parser.add_argument('--coverage', action='store_true',
                       help='Run with coverage reporting')
    parser.add_argument('--html-cov', action='store_true',
                       help='Generate HTML coverage report')
    
    # Pytest pass-through options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet output')
    parser.add_argument('--parallel', '-n', type=int, 
                       help='Number of parallel processes')
    parser.add_argument('--pattern', '-k', type=str,
                       help='Run tests matching pattern')
    parser.add_argument('--failed', '--lf', action='store_true',
                       help='Run only previously failed tests')
    parser.add_argument('--exitfirst', '-x', action='store_true',
                       help='Exit on first failure')
    
    # Additional options
    parser.add_argument('--install-deps', action='store_true',
                       help='Install test dependencies first')
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        print("Installing test dependencies...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements_test.txt'])
    
    # Build pytest command
    pytest_args = []
    
    # Test selection
    if args.unit:
        pytest_args.append('tests/unit')
    elif args.integration:
        pytest_args.append('tests/integration')
    elif args.performance:
        pytest_args.append('tests/performance')
    else:
        pytest_args.append('tests')
    
    # Output options
    if args.verbose:
        pytest_args.append('-v')
    elif args.quiet:
        pytest_args.append('-q')
    
    # Coverage options
    if args.coverage or args.html_cov:
        pytest_args.extend(['--cov=scripts', '--cov-report=term-missing'])
        if args.html_cov:
            pytest_args.append('--cov-report=html:htmlcov')
    
    # Speed options
    if args.fast:
        pytest_args.extend(['-m', 'not slow'])
    
    # Parallel execution
    if args.parallel:
        pytest_args.extend(['-n', str(args.parallel)])
    
    # Pattern matching
    if args.pattern:
        pytest_args.extend(['-k', args.pattern])
    
    # Failed tests only
    if args.failed:
        pytest_args.append('--lf')
    
    # Exit on first failure
    if args.exitfirst:
        pytest_args.append('-x')
    
    # Run tests
    return_code = run_pytest_command(pytest_args)
    
    # Summary
    if return_code == 0:
        print("\n✅ All tests passed!")
        if args.coverage or args.html_cov:
            print("📊 Coverage report generated")
            if args.html_cov:
                print("🔗 HTML coverage report: htmlcov/index.html")
    else:
        print(f"\n❌ Tests failed with return code {return_code}")
    
    return return_code

if __name__ == '__main__':
    sys.exit(main())