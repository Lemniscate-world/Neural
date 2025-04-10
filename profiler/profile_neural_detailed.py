#!/usr/bin/env python
"""
Detailed profiling of the Neural CLI startup time.
This script uses the cProfile module to get detailed information about what's taking time.
"""

import cProfile
import pstats
import io
import os
import sys

# Redirect stderr to /dev/null to suppress debug messages
stderr_backup = sys.stderr
sys.stderr = open(os.devnull, 'w')

# Profile the import of neural.cli
profiler = cProfile.Profile()
profiler.enable()

# Import neural.cli
import neural.cli

profiler.disable()

# Restore stderr
sys.stderr = stderr_backup

# Print the profiling results
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
ps.print_stats(30)  # Print top 30 functions by cumulative time
print(s.getvalue())
