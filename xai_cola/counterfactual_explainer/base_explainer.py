from abc import ABC, abstractmethod
import pandas as pd

from xai_cola.ml_model import Model
from xai_cola.data import PandasData

class CounterFactualExplainer(ABC):
    def __init__(self, ml_model:Model, data:PandasData, sample_num):
        """
        Initialize the CounterFactualExplainer class
        
        Parameters:
        model: Pre-trained model
        """
        self.ml_model = ml_model
        self.data = data
        self.sample_num = sample_num
        
        self.x_factual = data.get_x()
        self.target_name = data.get_target_name()
        
        self.factual, self.counterfactual = self.generate_counterfactuals()
        
    @abstractmethod
    def generate_counterfactuals(self) -> pd.DataFrame:
        """
        Generate counterfactuals for the given data
        
        Parameters:
        data: DataFrame type data variable
        
        Returns:
        DataFrame type counterfactual results
        """
        pass
