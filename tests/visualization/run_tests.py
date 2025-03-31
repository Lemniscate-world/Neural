#!/usr/bin/env python3
"""
Run all visualization tests.
"""

import os
import sys
import pytest

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

if __name__ == '__main__':
    # Get the directory of this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all test files in the directory
    test_files = [f for f in os.listdir(test_dir) if f.startswith('test_') and f.endswith('.py')]
    
    # Run the tests
    exit_code = pytest.main(['-xvs'] + [os.path.join(test_dir, f) for f in test_files])
    
    # Exit with the pytest exit code
    sys.exit(exit_code)
