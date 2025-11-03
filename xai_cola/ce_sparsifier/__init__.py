"""
CE Sparsifier - Counterfactual Explanation Sparsifier

This module contains the core COLA functionality for refining counterfactual explanations.
"""

# Import main class from ce_sparsifier
from .cola import COLA
from . import data
from . import models

__all__ = ['COLA', 'data', 'models']

