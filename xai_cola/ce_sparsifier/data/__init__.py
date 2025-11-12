"""
COLA Data Module

Usage:
    from xai_cola.data import COLAData
    
    # Initialize with DataFrame
    data = COLAData(factual_data=df, label_column='Risk')
    
    # Or with numpy array
    data = COLAData(factual_data=np_array, label_column='Risk', column_names=columns)
"""

from .coladata import COLAData

__all__ = ['COLAData']