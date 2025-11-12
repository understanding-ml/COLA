"""
Utilities for suppressing non-critical warnings in COLA

This module provides functions to suppress common warnings that don't affect
the functionality of COLA but can clutter the output.
"""

import os
import warnings


def suppress_joblib_warnings():
    """
    Suppress joblib/loky warnings about physical CPU core detection on Windows.
    
    This warning appears because Windows lacks certain system tools that 
    joblib/loky uses to detect physical CPU cores. The library automatically
    falls back to logical cores, which works fine for most use cases.
    
    You can set LOKY_MAX_CPU_COUNT to specify the number of cores manually.
    """
    # Set environment variable to suppress the warning
    # If not set, joblib will detect logical cores but show a warning
    # Setting it to None/empty means "use auto-detection" but without warning
    if 'LOKY_MAX_CPU_COUNT' not in os.environ:
        # Get logical core count (this is what joblib will use anyway)
        try:
            import multiprocessing
            logical_cores = multiprocessing.cpu_count()
            # Set to logical cores to suppress warning while maintaining functionality
            os.environ['LOKY_MAX_CPU_COUNT'] = str(logical_cores)
        except Exception:
            # If we can't detect, just set a reasonable default
            os.environ['LOKY_MAX_CPU_COUNT'] = '4'


def suppress_pandas_warnings():
    """
    Suppress common pandas deprecation warnings.
    
    Specifically suppresses warnings about:
    - Future deprecation warnings from pandas
    - Replace downcasting behavior (fixed in our code, but suppressing for safety)
    """
    warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
    warnings.filterwarnings('ignore', message='.*downcasting.*', category=FutureWarning)


def suppress_all_warnings():
    """
    Suppress all non-critical warnings commonly seen when using COLA.
    
    This includes:
    - joblib/loky CPU detection warnings (Windows)
    - pandas deprecation warnings
    """
    suppress_joblib_warnings()
    suppress_pandas_warnings()


def setup_warnings(verbose: bool = True):
    """
    Setup warning suppression for COLA.
    
    Parameters:
    -----------
    verbose : bool
        If True, print a message about warnings being suppressed.
        Default: True
    """
    suppress_all_warnings()
    if verbose:
        print("✓ Warning suppression enabled (joblib/loky, pandas)")

