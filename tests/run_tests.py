#!/usr/bin/env python
"""
Run all unit tests with coverage report

Usage:
    python -m pytest tests/ -v --tb=short
    python -m pytest tests/ -v --tb=short --cov=topo_map --cov-report=html
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def discover_and_run_tests():
    """Discover and run all tests in the tests directory"""
    
    # Create test suite
    loader = unittest.TestLoader()
    
    # Discover tests
    test_dir = project_root / 'tests'
    suite = loader.discover(test_dir, pattern='test_*.py', top_level_dir=str(project_root))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    exit_code = discover_and_run_tests()
    sys.exit(exit_code)