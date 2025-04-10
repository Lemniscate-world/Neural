#!/usr/bin/env python
"""
Profile the startup time of the Neural CLI.
This script measures how long it takes to import various modules.
"""

import time
import sys

def time_import(module_name):
    """Time how long it takes to import a module."""
    start_time = time.time()
    try:
        __import__(module_name)
        end_time = time.time()
        return end_time - start_time, True
    except ImportError as e:
        end_time = time.time()
        return end_time - start_time, False

# List of modules to profile
modules_to_profile = [
    'neural',
    'neural.cli',
    'neural.cli_aesthetics',
    'tensorflow',
    'torch',
    'matplotlib',
    'graphviz',
    'lark',
    'dash',
    'plotly',
    'optuna',
    'jax',
]

print("Profiling Neural CLI startup time...")
print("-" * 50)
print(f"{'Module':<20} {'Time (s)':<10} {'Status':<10}")
print("-" * 50)

total_time = 0
for module in modules_to_profile:
    import_time, success = time_import(module)
    total_time += import_time
    status = "Success" if success else "Failed"
    print(f"{module:<20} {import_time:.4f}s    {status}")

print("-" * 50)
print(f"{'Total time':<20} {total_time:.4f}s")
print("-" * 50)
