"""
Version information for Neural CLI.
"""

import importlib.metadata

# Get version from package metadata
try:
    __version__ = importlib.metadata.version("neural")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
