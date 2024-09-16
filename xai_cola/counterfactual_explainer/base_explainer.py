from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from xai_cola.ml_model import Model
from xai_cola.data import BaseData

class CounterFactualExplainer(ABC):
    def __init__(self, ml_model:Model, data:BaseData=None):
        """
        Initialize the CounterFactualExplainer class
        
        Parameters:
        model: Pre-trained model
        data: use our wrapperred-data (NumpyData, PandasData, etc.)
        """ 
        self.ml_model = ml_model
        self.data = data                                               

        # if specific counterfactual algorithm requires data to initialize the explainer
        if data:
            self.x_factual_pandas = data.get_x()  # pandas type without 'target' column
            self.target_name = data.get_target_name()
        else:
            self.x_factual_pandas = None
            self.target_name = None


    def _process_data(self, data: BaseData = None):
        """
        shared data processing logic

        Parameters:
        data: fatual data(don't need target column)

        """
        if data:
            self.data = data
            self.x_factual_pandas = data.get_x()
            self.target_name = data.get_target_name()
        elif self.data is None:
            raise ValueError("Data must be provided either in __init__ or in generate_counterfactuals")

        
    @abstractmethod
    def generate_counterfactuals(self, data:BaseData=None) -> np.ndarray:
        """
        Generate counterfactuals for the given data
        
        Parameters:
        data: BaseData type, the factual data(don't need target column)
        params: parameters for specific counterfactual algorithm
        
        Returns:
        ndarray type counterfactual results
        """

        pass
