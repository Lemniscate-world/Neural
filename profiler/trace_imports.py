#!/usr/bin/env python
"""
Trace all imports made when importing neural.cli.
This helps identify which modules are being imported and in what order.
"""

import sys
import os
from types import ModuleType

# Original import function
original_import = __import__

# Keep track of imported modules
imported_modules = []

# Redirect stderr to /dev/null to suppress debug messages
stderr_backup = sys.stderr
sys.stderr = open(os.devnull, 'w')

# Custom import function that logs imports
def custom_import(name, *args, **kwargs):
    if name not in imported_modules:
        imported_modules.append(name)
        print(f"Importing: {name}")
    return original_import(name, *args, **kwargs)

# Replace the built-in import function
sys.__import__ = custom_import

# Import neural.cli
try:
    import neural.cli
except Exception as e:
    print(f"Error importing neural.cli: {e}")

# Restore the original import function
sys.__import__ = original_import

# Restore stderr
sys.stderr = stderr_backup

# Print summary
print("\nSummary of imports:")
print(f"Total modules imported: {len(imported_modules)}")

# Group imports by category
ml_frameworks = [m for m in imported_modules if m.startswith(('tensorflow', 'torch', 'jax', 'optuna'))]
visualization = [m for m in imported_modules if m.startswith(('matplotlib', 'plotly', 'dash', 'graphviz'))]
neural_modules = [m for m in imported_modules if m.startswith('neural')]

print(f"\nML Frameworks ({len(ml_frameworks)}):")
for m in ml_frameworks[:10]:  # Show only first 10
    print(f"  - {m}")
if len(ml_frameworks) > 10:
    print(f"  - ... and {len(ml_frameworks) - 10} more")

print(f"\nVisualization Libraries ({len(visualization)}):")
for m in visualization[:10]:
    print(f"  - {m}")
if len(visualization) > 10:
    print(f"  - ... and {len(visualization) - 10} more")

print(f"\nNeural Modules ({len(neural_modules)}):")
for m in neural_modules:
    print(f"  - {m}")
