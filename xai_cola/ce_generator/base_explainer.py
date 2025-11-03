from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from xai_cola.ce_sparsifier.models import Model
from xai_cola.ce_sparsifier.data import COLAData

class CounterFactualExplainer(ABC):
    def __init__(self, ml_model, data:COLAData=None):
        """
        Initialize the CounterFactualExplainer class
        
        Parameters:
        -----------
        ml_model : Model
            Pre-trained model wrapped in Model interface
        data : COLAData, optional
            Data wrapper containing factual data
        """ 
        self.ml_model = ml_model
        self.data = data                                               

        # if specific counterfactual algorithm requires data to initialize the explainer
        if data:
            # Extract features only (without target column)
            self.x_factual_pandas = data.get_factual_features()
            self.target_name = data.get_label_column()
        else:
            self.x_factual_pandas = None
            self.target_name = None


    def _process_data(self, data: COLAData = None):
        """
        Shared data processing logic

        Parameters:
        -----------
        data : COLAData
            Factual data wrapper
        """
        if data:
            self.data = data
            self.x_factual_pandas = data.get_factual_features()
            self.target_name = data.get_label_column()
        elif self.data is None:
            raise ValueError("Data must be provided either in __init__ or in generate_counterfactuals")

        
    @abstractmethod
    def generate_counterfactuals(self, data:COLAData=None) -> np.ndarray:
        """
        Generate counterfactuals for the given data
        
        Parameters:
        -----------
        data : COLAData
            Data wrapper containing factual data
        
        Returns:
        --------
        ndarray
            Counterfactual results
        """
        pass
