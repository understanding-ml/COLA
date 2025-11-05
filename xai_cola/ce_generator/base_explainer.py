from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from xai_cola.ce_sparsifier.models import Model
from xai_cola.ce_sparsifier.data import COLAData

class CounterFactualExplainer(ABC):
    def __init__(self, ml_model):
        """
        Initialize the CounterFactualExplainer class

        Parameters:
        -----------
        ml_model : Model
            Pre-trained model wrapped in Model interface
        """
        self.ml_model = ml_model
        self.data = None
        self.x_factual_pandas = None
        self.target_name = None


    def _process_data(self, data: COLAData):
        """
        Shared data processing logic

        Parameters:
        -----------
        data : COLAData
            Factual data wrapper (required)

        Raises:
        -------
        ValueError
            If data is None
        """
        if data is None:
            raise ValueError("Data must be provided in generate_counterfactuals method")

        self.data = data
        self.x_factual_pandas = data.get_factual_features()
        self.target_name = data.get_label_column()

        
    @abstractmethod
    def generate_counterfactuals(self, data: COLAData, **kwargs) -> np.ndarray:
        """
        Generate counterfactuals for the given data

        Parameters:
        -----------
        data : COLAData
            Data wrapper containing factual data (required)
        **kwargs : dict
            Additional algorithm-specific parameters

        Returns:
        --------
        ndarray or tuple
            Counterfactual results
        """
        pass
