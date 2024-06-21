from abc import ABC, abstractmethod
import pandas as pd

class CounterFactualExplainer(ABC):
    def __init__(self, ml_model, x_factual: pd.DataFrame):
        """
        Initialize the CounterFactualExplainer class
        
        Parameters:
        model: Pre-trained model
        """
        self.ml_model = ml_model
        self.x_factual = x_factual
    # @abstractmethod
    def generate_counterfactuals(self) -> pd.DataFrame:
        """
        Generate counterfactuals for the given data
        
        Parameters:
        data: DataFrame type data variable
        
        Returns:
        DataFrame type counterfactual results
        """
        pass
