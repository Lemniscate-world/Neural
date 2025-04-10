"""
Lazy imports for Neural CLI.
This module provides lazy loading for heavy dependencies.
"""

import importlib
import sys
import os
import time
import logging
import warnings

# Configure environment to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages
os.environ['PYTHONWARNINGS'] = 'ignore'   # Suppress Python warnings
os.environ['MPLBACKEND'] = 'Agg'          # Non-interactive matplotlib backend

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LazyLoader:
    """
    Lazily import a module only when it's actually needed.
    This helps reduce startup time by deferring expensive imports.
    """
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None
        self._cached_attrs = {}

    def __getattr__(self, name):
        # Check if we've already cached this attribute
        if name in self._cached_attrs:
            return self._cached_attrs[name]

        # If the module hasn't been loaded yet, load it
        if self.module is None:
            # Temporarily redirect stderr to suppress warnings during import
            original_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')

            try:
                start_time = time.time()
                self.module = importlib.import_module(self.module_name)
                end_time = time.time()
                logger.debug(f"Lazy-loaded {self.module_name} in {end_time - start_time:.2f} seconds")
            finally:
                # Restore stderr
                sys.stderr.close()
                sys.stderr = original_stderr

        # Get the attribute and cache it for future use
        attr = getattr(self.module, name)
        self._cached_attrs[name] = attr
        return attr

# Define lazy loaders for heavy dependencies
def lazy_import(module_name):
    """Create a lazy loader for a module."""
    return LazyLoader(module_name)

# Create lazy loaders for heavy dependencies
tensorflow = lazy_import('tensorflow')
torch = lazy_import('torch')
jax = lazy_import('jax')
matplotlib = lazy_import('matplotlib')
plotly = lazy_import('plotly')
dash = lazy_import('dash')
optuna = lazy_import('optuna')

# Create lazy loaders for Neural modules that depend on heavy dependencies
shape_propagator = lazy_import('neural.shape_propagation.shape_propagator')
tensor_flow = lazy_import('neural.dashboard.tensor_flow')
hpo = lazy_import('neural.hpo.hpo')
code_generator = lazy_import('neural.code_generation.code_generator')

# Function to get a module from a lazy loader
def get_module(lazy_loader):
    """Get the actual module from a lazy loader."""
    if isinstance(lazy_loader, LazyLoader):
        if lazy_loader.module is None:
            start_time = time.time()
            lazy_loader.module = importlib.import_module(lazy_loader.module_name)
            end_time = time.time()
            logger.debug(f"Lazy-loaded {lazy_loader.module_name} in {end_time - start_time:.2f} seconds")
        return lazy_loader.module
    return lazy_loader
