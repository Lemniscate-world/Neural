#!/usr/bin/env python
"""
Alternative approach to trace imports in Neural CLI.
"""

import os
import sys
import time
import importlib

# Redirect stderr to /dev/null to suppress debug messages
stderr_backup = sys.stderr
sys.stderr = open(os.devnull, 'w')

# Set environment variables to suppress debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

# List of modules to check
modules_to_check = [
    'tensorflow',
    'torch',
    'jax',
    'matplotlib',
    'plotly',
    'dash',
    'graphviz',
    'optuna',
    'lark',
    'numpy',
    'networkx',
]

# Check if these modules are imported when importing neural.cli
print("Checking which modules are imported by neural.cli...")
print("-" * 60)

# First, get the currently loaded modules
before_modules = set(sys.modules.keys())

# Import neural.cli
start_time = time.time()
import neural.cli
end_time = time.time()

# Get the modules loaded after importing neural.cli
after_modules = set(sys.modules.keys())

# Find the new modules that were loaded
new_modules = after_modules - before_modules

# Check which of our target modules were loaded
print(f"{'Module':<15} {'Imported':<10} {'Present in sys.modules':<20}")
print("-" * 60)
for module in modules_to_check:
    # Check if the module or any submodule was imported
    module_imported = any(m.startswith(module + '.') or m == module for m in new_modules)
    module_in_sys = module in sys.modules
    print(f"{module:<15} {'Yes' if module_imported else 'No':<10} {'Yes' if module_in_sys else 'No':<20}")

print("-" * 60)
print(f"Total time to import neural.cli: {end_time - start_time:.2f} seconds")
print(f"Total new modules loaded: {len(new_modules)}")

# Restore stderr
sys.stderr = stderr_backup

# Print some of the new modules for inspection
print("\nSample of new modules loaded:")
for module in sorted(list(new_modules))[:20]:
    print(f"  - {module}")

if len(new_modules) > 20:
    print(f"  - ... and {len(new_modules) - 20} more")

# Check if any of the heavy ML frameworks are imported at the top level
print("\nChecking for heavy imports at the top level of neural.cli...")
with open(neural.cli.__file__, 'r') as f:
    cli_code = f.read()

import_lines = [line.strip() for line in cli_code.split('\n') if line.strip().startswith('import ') or line.strip().startswith('from ')]
print("\nImport statements in neural.cli:")
for line in import_lines[:20]:
    print(f"  {line}")
if len(import_lines) > 20:
    print(f"  ... and {len(import_lines) - 20} more")
