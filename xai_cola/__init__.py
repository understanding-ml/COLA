"""
COLA - COunterfactual explanations with Limited Actions

A Python library for refining counterfactual explanations with action-limited constraints.

Main modules:
    - ce_generator: Counterfactual explanation generators (DiCE, DisCount, WachterCF, etc.)
    - ce_sparsifier: COLA sparsification functionality (policies, matching, attribution)

Usage:
    >>> from xai_cola.ce_generator import DiCE, DisCount, WachterCF
    >>> from xai_cola.ce_sparsifier import COLA
    >>> from xai_cola.ce_sparsifier.data import COLAData
    >>> from xai_cola.ce_sparsifier.models import Model
"""

# Import version
from .version import __version__

# Import from ce_sparsifier for backward compatibility
from .ce_sparsifier import COLA
from .ce_sparsifier import data
from .ce_sparsifier import models

# Re-export for backward compatibility
__all__ = [
    'COLA',
    'data',
    'models',
    '__version__',
]
